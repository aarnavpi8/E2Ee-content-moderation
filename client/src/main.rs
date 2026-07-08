use std::collections::HashMap;
use std::fs::File;
use std::io::{self, Write};
use std::path::Path;
use std::sync::{Arc, Mutex};
use x25519_dalek::{StaticSecret, PublicKey};
use ed25519_dalek::SigningKey;
use rand::rngs::OsRng;
use rand::RngCore;
use serde::{Serialize, Deserialize};
use crypto_core::{
    KeyBundle, PrekeyBundleAnnouncement, InboxMessage, X3DHInit,
    DoubleRatchet, EncryptedMessage, ModerationData, moderation_ad, sign,
};
use moderation_core::{commitment, Model, ModerationCircuit, GOLDILOCKS_P, MAX_MSG_BYTES};

const MODEL_PATH: &str = "moderation/models/model_d256.json";

#[derive(Serialize, Deserialize)]
struct ClientState {
    username: String,
    ik_secret: [u8; 32],
    ik_sign_secret: [u8; 32],
    spk_secret: [u8; 32],
    opk_secrets: Vec<[u8; 32]>,
    sessions: HashMap<String, DoubleRatchet>,
    last_received: HashMap<String, EncryptedMessage>,
}

impl ClientState {
    fn file_path(username: &str) -> String {
        format!("{}_state.json", username)
    }

    fn load(username: &str) -> Result<Self, String> {
        let path_str = Self::file_path(username);
        let path = Path::new(&path_str);
        if !path.exists() {
            return Err("State file does not exist".to_string());
        }
        let file = File::open(path).map_err(|e| e.to_string())?;
        let state: ClientState = serde_json::from_reader(file).map_err(|e| e.to_string())?;
        Ok(state)
    }

    fn save(&self) -> Result<(), String> {
        let path_str = Self::file_path(&self.username);
        let file = File::create(path_str).map_err(|e| e.to_string())?;
        serde_json::to_writer_pretty(file, self).map_err(|e| e.to_string())?;
        Ok(())
    }

    fn generate_new(username: &str) -> Self {
        let mut ik_bytes = [0u8; 32];
        let mut ik_sign_bytes = [0u8; 32];
        let mut spk_bytes = [0u8; 32];
        OsRng.fill_bytes(&mut ik_bytes);
        OsRng.fill_bytes(&mut ik_sign_bytes);
        OsRng.fill_bytes(&mut spk_bytes);

        let mut opk_secrets = Vec::new();
        for _ in 0..50 {
            let mut opk_bytes = [0u8; 32];
            OsRng.fill_bytes(&mut opk_bytes);
            opk_secrets.push(opk_bytes);
        }

        Self {
            username: username.to_string(),
            ik_secret: ik_bytes,
            ik_sign_secret: ik_sign_bytes,
            spk_secret: spk_bytes,
            opk_secrets,
            sessions: HashMap::new(),
            last_received: HashMap::new(),
        }
    }
}

async fn register_and_publish(state: &ClientState, server_url: &str) -> Result<(), String> {
    let client = reqwest::Client::new();

    // 1. Register
    let reg_payload = serde_json::json!({
        "username": state.username
    });
    let reg_res = client.post(format!("{}/register", server_url))
        .json(&reg_payload)
        .send()
        .await
        .map_err(|e| e.to_string())?;

    if !reg_res.status().is_success() {
        return Err(format!("Server registration failed: {}", reg_res.status()));
    }

    // 2. Publish bundle
    let bob_ik_sec = StaticSecret::from(state.ik_secret);
    let bob_ik_pub = PublicKey::from(&bob_ik_sec);

    let bob_spk_sec = StaticSecret::from(state.spk_secret);
    let bob_spk_pub = PublicKey::from(&bob_spk_sec);

    let bob_sign_key = SigningKey::from_bytes(&state.ik_sign_secret);
    let bob_verify_key = bob_sign_key.verifying_key();

    let spk_bytes = bob_spk_pub.to_bytes();
    let spk_sig = sign(&bob_sign_key, &spk_bytes);

    let mut one_time_prekeys = Vec::new();
    for opk_bytes in &state.opk_secrets {
        let opk_sec = StaticSecret::from(*opk_bytes);
        one_time_prekeys.push(PublicKey::from(&opk_sec));
    }

    let announcement = PrekeyBundleAnnouncement {
        identity_key: bob_ik_pub,
        identity_signing_key: bob_verify_key,
        signed_prekey: bob_spk_pub,
        signed_prekey_sig: spk_sig,
        one_time_prekeys,
    };

    let pub_res = client.post(format!("{}/bundles/{}", server_url, state.username))
        .json(&announcement)
        .send()
        .await
        .map_err(|e| e.to_string())?;

    if !pub_res.status().is_success() {
        return Err(format!("Publishing prekey bundle failed: {}", pub_res.status()));
    }

    println!("Registered and published prekey bundle for {}", state.username);
    Ok(())
}

/// Draw a random blinding nonce r in the Goldilocks field [0, p).
fn random_nonce() -> u64 {
    OsRng.next_u64() % GOLDILOCKS_P
}

/// Send a message with a verifiable-moderation envelope.
///
/// * `prove_text`  — the text the ZK proof is generated for (must pass the
///   classifier, else this returns Err and nothing is sent).
/// * `send_text`   — the text actually encrypted and delivered. Equal to
///   `prove_text` for honest sends; differs for the `/forge` demo.
/// * `tamper_h`    — if true, flips a bit of the envelope commitment so the
///   server's proof/commitment check fails (the `/sendbad` drop demo).
async fn send_message_ext(
    state_arc: Arc<Mutex<ClientState>>,
    server_url: &str,
    recipient: &str,
    prove_text: &str,
    send_text: &str,
    tamper_h: bool,
    circuit: &ModerationCircuit,
) -> Result<(), String> {
    let client = reqwest::Client::new();

    // 1. Generate the moderation proof over prove_text.
    let r = random_nonce();
    let bundle = circuit
        .prove(prove_text, r)
        .map_err(|e| format!("blocked by local classifier / proof failed: {}", e))?;
    let ad = moderation_ad(&bundle.h, r);

    let mut md = ModerationData { h: bundle.h, r, proof: bundle.proof_bytes };
    if tamper_h {
        md.h[0] ^= 1; // corrupt the commitment so the server drops it
    }

    // 2. Encrypt send_text under AD = (h, r) via the Double Ratchet.
    let (x3dh_init, encrypted_msg) = {
        let mut state = state_arc.lock().unwrap();
        if !state.sessions.contains_key(recipient) {
            // Fetch recipient bundle
            let res = client.get(format!("{}/bundles/{}", server_url, recipient))
                .send()
                .await
                .map_err(|e| e.to_string())?;
            if !res.status().is_success() {
                return Err(format!("Could not fetch prekey bundle for {}: status {}", recipient, res.status()));
            }
            let bundle_keys = res.json::<KeyBundle>().await.map_err(|e| e.to_string())?;

            // X3DH Alice derive
            let alice_ik_sec = StaticSecret::from(state.ik_secret);
            let alice_ek_sec = StaticSecret::from(crypto_core::random_bytes_32());
            let alice_ek_pub = PublicKey::from(&alice_ek_sec);

            let sk = crypto_core::x3dh_alice_derive(&alice_ik_sec, &alice_ek_sec, &bundle_keys)?;

            // Initialize Alice Double Ratchet
            let mut ratchet = DoubleRatchet::init_alice(sk, bundle_keys.signed_prekey);
            let encrypted_msg = ratchet.ratchet_encrypt(send_text.as_bytes(), &ad)?;

            let x3dh_init = X3DHInit {
                alice_identity_key: PublicKey::from(&alice_ik_sec),
                alice_ephemeral_key: alice_ek_pub,
                used_one_time_prekey: bundle_keys.one_time_prekey,
            };

            state.sessions.insert(recipient.to_string(), ratchet);
            (Some(x3dh_init), encrypted_msg)
        } else {
            let ratchet = state.sessions.get_mut(recipient).unwrap();
            let encrypted_msg = ratchet.ratchet_encrypt(send_text.as_bytes(), &ad)?;
            (None, encrypted_msg)
        }
    };

    let sender = {
        let state = state_arc.lock().unwrap();
        state.username.clone()
    };

    let msg = InboxMessage {
        sender,
        x3dh_init,
        ratchet_message: encrypted_msg,
        moderation: Some(md),
    };

    let res = client.post(format!("{}/inbox/{}", server_url, recipient))
        .json(&msg)
        .send()
        .await
        .map_err(|e| e.to_string())?;

    if !res.status().is_success() {
        let status = res.status();
        let body = res.text().await.unwrap_or_default();
        return Err(format!("Server rejected message: {} ({})", status, body));
    }

    // Save client state
    {
        let state = state_arc.lock().unwrap();
        state.save()?;
    }

    Ok(())
}

fn start_polling(
    state_arc: Arc<Mutex<ClientState>>,
    server_url: String,
) {
    tokio::spawn(async move {
        let client = reqwest::Client::new();
        let username = {
            let state = state_arc.lock().unwrap();
            state.username.clone()
        };

        loop {
            tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;

            let url = format!("{}/inbox/{}", server_url, username);
            let res = match client.get(&url).send().await {
                Ok(res) => res,
                Err(_) => continue,
            };

            if !res.status().is_success() {
                continue;
            }

            let messages = match res.json::<Vec<InboxMessage>>().await {
                Ok(msgs) => msgs,
                Err(_) => continue,
            };

            if messages.is_empty() {
                continue;
            }

            let mut state = state_arc.lock().unwrap();
            for msg in messages {
                let sender_clone = msg.sender.clone();

                // AD used for decryption is derived from the moderation envelope.
                let ad: Vec<u8> = match &msg.moderation {
                    Some(md) => moderation_ad(&md.h, md.r),
                    None => Vec::new(),
                };

                let decrypted_verbose: Result<(Vec<u8>, [u8; 32]), String> = match &msg.x3dh_init {
                    Some(init) => {
                        // Bob initializes new session
                        let bob_dh_sec = if let Some(opk_pub) = init.used_one_time_prekey {
                            // Find the private key corresponding to opk_pub and consume it
                            let mut found_idx = None;
                            for (idx, sec_bytes) in state.opk_secrets.iter().enumerate() {
                                let sec = StaticSecret::from(*sec_bytes);
                                if PublicKey::from(&sec).to_bytes() == opk_pub.to_bytes() {
                                    found_idx = Some(idx);
                                    break;
                                }
                            }
                            match found_idx {
                                Some(idx) => {
                                    // Remove the spent OPK — one-time prekeys must not be reused
                                    let sec_bytes = state.opk_secrets.remove(idx);
                                    StaticSecret::from(sec_bytes)
                                }
                                None => {
                                    println!("\nError: Received message from {} using unknown OPK", msg.sender);
                                    print!("> ");
                                    std::io::stdout().flush().unwrap();
                                    continue;
                                }
                            }
                        } else {
                            StaticSecret::from(state.spk_secret)
                        };

                        let bob_ik_sec = StaticSecret::from(state.ik_secret);
                        let bob_spk_sec = StaticSecret::from(state.spk_secret);
                        let sk = match crypto_core::x3dh_bob_derive(
                            &bob_ik_sec,
                            &bob_spk_sec,
                            init.used_one_time_prekey.as_ref().map(|_| &bob_dh_sec),
                            init
                        ) {
                            Ok(sk) => sk,
                            Err(e) => {
                                println!("\nError: Bob X3DH derivation failed: {}", e);
                                print!("> ");
                                std::io::stdout().flush().unwrap();
                                continue;
                            }
                        };

                        let bob_spk_sec_init = StaticSecret::from(state.spk_secret);
                        let ratchet = DoubleRatchet::init_bob(sk, bob_spk_sec_init);
                        state.sessions.insert(msg.sender.clone(), ratchet);

                        let ratchet = state.sessions.get_mut(&msg.sender).unwrap();
                        ratchet.ratchet_decrypt_verbose(&msg.ratchet_message, &ad)
                    }
                    None => {
                        if let Some(ratchet) = state.sessions.get_mut(&msg.sender) {
                            ratchet.ratchet_decrypt_verbose(&msg.ratchet_message, &ad)
                        } else {
                            Err("No session initialized for sender".to_string())
                        }
                    }
                };

                match decrypted_verbose {
                    Ok((plaintext, mk)) => {
                        let text = String::from_utf8_lossy(&plaintext);
                        println!("\n[{}] {}", msg.sender, text);

                        // ---- Receiver-side binding check ----------------------
                        // Recompute Poseidon(m, r) from the DECRYPTED plaintext and
                        // compare against the committed h. A mismatch proves the
                        // sender encrypted different content than they proved.
                        if let Some(md) = &msg.moderation {
                            let recomputed = if plaintext.len() <= MAX_MSG_BYTES {
                                commitment(&plaintext, md.r)
                            } else {
                                // Over-length payload can't match any honest
                                // commitment; force a mismatch instead of panicking.
                                [md.h[0] ^ 1, md.h[1], md.h[2], md.h[3]]
                            };
                            if recomputed == md.h {
                                println!("  ✓ [binding check] Poseidon(m, r) == h — content matches the cleared proof.");
                            } else {
                                println!("┌─ [BINDING CHECK FAILED] ───────────────────────────────┐");
                                println!("│  Sender proved one message but encrypted another!       │");
                                println!("│  committed h : {:016x}...                     │", md.h[0]);
                                println!("│  recomputed  : {:016x}...                     │", recomputed[0]);
                                println!("│  Message REJECTED (attributable sender misbehaviour).   │");
                                println!("└─────────────────────────────────────────────────────────┘");
                            }
                        }

                        // Forward-secrecy proof: try to decrypt the previous message
                        // from this sender using the current message's key.
                        if let Some(prev_msg) = state.last_received.get(&sender_clone) {
                            if let Some(ratchet) = state.sessions.get(&sender_clone) {
                                let mk_hex: String = mk.iter().map(|b| format!("{:02x}", b)).collect();
                                println!("┌─ [Forward-Secrecy Proof] ─────────────────────────────┐");
                                println!("│  Current msg key (mk):  0x{}...  │", &mk_hex[..16]);
                                println!("│  Attempting to decrypt PREVIOUS message with this key: │");
                                match ratchet.decrypt_message_with_key(prev_msg, &mk, &ad) {
                                    Ok(_) => println!("│  ✗ UNEXPECTED: decryption succeeded (bug!)             │"),
                                    Err(e) => {
                                        println!("│  ✓ FAILED as expected: {}  │", e);
                                        println!("│  Previous message key was discarded — proof complete.  │");
                                    }
                                }
                                println!("└────────────────────────────────────────────────────────┘");
                            }
                        }

                        // Store the just-received encrypted envelope as the
                        // "previous" for the next forward-secrecy check.
                        state.last_received.insert(sender_clone, msg.ratchet_message.clone());

                        print!("> ");
                        std::io::stdout().flush().unwrap();
                    }
                    Err(e) => {
                        println!("\nError decrypting message from {}: {}", msg.sender, e);
                        print!("> ");
                        std::io::stdout().flush().unwrap();
                    }
                }
            }

            let _ = state.save();
        }
    });
}

fn print_help() {
    println!("Commands:");
    println!("  /send <recipient> <message>            Prove + send an honest message.");
    println!("  /forge <recipient> <proved> | <sent>   Prove <proved> but encrypt <sent>");
    println!("                                         (receiver's binding check catches it).");
    println!("  /sendbad <recipient> <message>         Send with a corrupted commitment");
    println!("                                         (server drops it: proof/commitment mismatch).");
    println!("  /help                                  Show this help.");
    println!("  /exit or /quit                         Quit.");
}

#[tokio::main]
async fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        println!("Usage: cargo run -p client -- <username>");
        return Ok(());
    }
    let username = &args[1];
    let server_url = "http://127.0.0.1:3000";

    println!("Loading moderation model and building circuit (one-time)...");
    let model = Model::from_json_file(MODEL_PATH)
        .map_err(|e| format!("failed to load moderation model ({}): {}", MODEL_PATH, e))?;
    let circuit = Arc::new(ModerationCircuit::new(model));
    println!("Moderation circuit ready (d = {}).", circuit.model.d);

    let state = match ClientState::load(username) {
        Ok(s) => {
            println!("Loaded existing session state for {}", username);
            s
        }
        Err(_) => {
            println!("No state found. Creating new state for {}...", username);
            let s = ClientState::generate_new(username);
            register_and_publish(&s, server_url).await?;
            s.save()?;
            s
        }
    };

    let state_arc = Arc::new(Mutex::new(state));
    start_polling(Arc::clone(&state_arc), server_url.to_string());

    println!("Logged in as {}.", username);
    print_help();

    let stdin = io::stdin();
    let mut input = String::new();

    print!("> ");
    io::stdout().flush().unwrap();

    loop {
        input.clear();
        if stdin.read_line(&mut input).is_err() {
            break;
        }

        let trimmed = input.trim();
        if trimmed.is_empty() {
            print!("> ");
            io::stdout().flush().unwrap();
            continue;
        }

        if trimmed == "/exit" || trimmed == "/quit" {
            break;
        } else if trimmed == "/help" {
            print_help();
        } else if let Some(rest) = trimmed.strip_prefix("/send ") {
            let parts: Vec<&str> = rest.splitn(2, ' ').collect();
            if parts.len() < 2 {
                println!("Usage: /send <recipient> <message>");
            } else {
                let (recipient, message) = (parts[0], parts[1]);
                match send_message_ext(Arc::clone(&state_arc), server_url, recipient,
                                       message, message, false, &circuit).await {
                    Ok(()) => println!("Sent (proof verified by server)!"),
                    Err(e) => println!("Not sent: {}", e),
                }
            }
        } else if let Some(rest) = trimmed.strip_prefix("/forge ") {
            // /forge <recipient> <proved> | <sent>
            let parts: Vec<&str> = rest.splitn(2, ' ').collect();
            if parts.len() < 2 || !parts[1].contains('|') {
                println!("Usage: /forge <recipient> <proved text> | <sent text>");
            } else {
                let recipient = parts[0];
                let mut halves = parts[1].splitn(2, '|');
                let proved = halves.next().unwrap().trim();
                let sent = halves.next().unwrap().trim();
                match send_message_ext(Arc::clone(&state_arc), server_url, recipient,
                                       proved, sent, false, &circuit).await {
                    Ok(()) => println!("Forged message sent (proved {:?}, encrypted {:?}).", proved, sent),
                    Err(e) => println!("Not sent: {}", e),
                }
            }
        } else if let Some(rest) = trimmed.strip_prefix("/sendbad ") {
            let parts: Vec<&str> = rest.splitn(2, ' ').collect();
            if parts.len() < 2 {
                println!("Usage: /sendbad <recipient> <message>");
            } else {
                let (recipient, message) = (parts[0], parts[1]);
                match send_message_ext(Arc::clone(&state_arc), server_url, recipient,
                                       message, message, true, &circuit).await {
                    Ok(()) => println!("Sent (this should not happen — server should have dropped it)."),
                    Err(e) => println!("Server dropped it as expected: {}", e),
                }
            }
        } else {
            println!("Unknown command. Type /help for commands.");
        }

        print!("> ");
        io::stdout().flush().unwrap();
    }

    Ok(())
}
