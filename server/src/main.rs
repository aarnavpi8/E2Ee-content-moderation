// NOTE: This server is a dumb store-and-forward relay for confidentiality — it
// only stores public keys, prekey bundles, and opaque ciphertext payloads, and
// never handles private keys or plaintexts.
//
// It additionally acts as a ZERO-KNOWLEDGE MIDDLEBOX for content moderation:
// before relaying a message it verifies the attached Plonky2 proof that the
// committed plaintext passes the public classifier. It gates purely on proof
// validity + commitment match; it never decrypts and never sees the plaintext.

use axum::{
    routing::post,
    Router,
    extract::{Path, State},
    Json,
    http::StatusCode,
};
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};
use crypto_core::{KeyBundle, PrekeyBundleAnnouncement, InboxMessage};
use moderation_core::{Model, ModerationCircuit};

const MODEL_PATH: &str = "moderation/models/model_d256.json";

struct ServerState {
    users: HashSet<String>,
    bundles: HashMap<String, PrekeyBundleAnnouncement>,
    inboxes: HashMap<String, Vec<InboxMessage>>,
}

#[derive(Clone)]
struct AppState {
    inner: Arc<Mutex<ServerState>>,
    circuit: Arc<ModerationCircuit>,
}

#[derive(serde::Deserialize)]
struct RegisterRequest {
    username: String,
}

async fn register(
    State(state): State<AppState>,
    Json(payload): Json<RegisterRequest>,
) -> Result<StatusCode, (StatusCode, String)> {
    let mut state = state.inner.lock().unwrap();
    if state.users.contains(&payload.username) {
        return Err((StatusCode::BAD_REQUEST, "Username already registered".to_string()));
    }
    state.users.insert(payload.username.clone());
    state.inboxes.insert(payload.username, Vec::new());
    Ok(StatusCode::OK)
}

async fn publish_bundle(
    Path(user): Path<String>,
    State(state): State<AppState>,
    Json(announcement): Json<PrekeyBundleAnnouncement>,
) -> Result<StatusCode, (StatusCode, String)> {
    let mut state = state.inner.lock().unwrap();
    if !state.users.contains(&user) {
        return Err((StatusCode::NOT_FOUND, "User not found".to_string()));
    }
    state.bundles.insert(user, announcement);
    Ok(StatusCode::OK)
}

async fn fetch_bundle(
    Path(user): Path<String>,
    State(state): State<AppState>,
) -> Result<Json<KeyBundle>, (StatusCode, String)> {
    let mut state = state.inner.lock().unwrap();
    let announcement = state.bundles.get_mut(&user)
        .ok_or_else(|| (StatusCode::NOT_FOUND, "Prekey bundle not found for user".to_string()))?;

    // Consume and delete one One-Time Prekey (OPK) from the pool to prevent reuse
    let opk = announcement.one_time_prekeys.pop();

    let bundle = KeyBundle {
        identity_key: announcement.identity_key,
        identity_signing_key: announcement.identity_signing_key,
        signed_prekey: announcement.signed_prekey,
        signed_prekey_sig: announcement.signed_prekey_sig,
        one_time_prekey: opk,
    };

    Ok(Json(bundle))
}

async fn enqueue_message(
    Path(user): Path<String>,
    State(state): State<AppState>,
    Json(msg): Json<InboxMessage>,
) -> Result<StatusCode, (StatusCode, String)> {
    // ---- Zero-knowledge moderation gate --------------------------------
    // The message is relayed ONLY if it carries a valid proof whose committed
    // hash matches the h in the envelope. No plaintext is ever inspected.
    let verdict = match &msg.moderation {
        Some(md) => state.circuit.verify(&md.proof, &md.h),
        None => false, // un-moderated messages are not relayed
    };
    if !verdict {
        println!("[moderation] DROPPED message for {} from {}: proof invalid or missing",
                 user, msg.sender);
        return Err((StatusCode::FORBIDDEN, "Moderation proof invalid or missing".to_string()));
    }
    println!("[moderation] ACCEPTED message for {} from {}: proof verified", user, msg.sender);

    let mut inner = state.inner.lock().unwrap();
    if !inner.users.contains(&user) {
        return Err((StatusCode::NOT_FOUND, "Target user not found".to_string()));
    }

    let inbox = inner.inboxes.entry(user).or_insert_with(Vec::new);
    inbox.push(msg);
    Ok(StatusCode::OK)
}

async fn drain_inbox(
    Path(user): Path<String>,
    State(state): State<AppState>,
) -> Result<Json<Vec<InboxMessage>>, (StatusCode, String)> {
    let mut inner = state.inner.lock().unwrap();
    if !inner.users.contains(&user) {
        return Err((StatusCode::NOT_FOUND, "User not found".to_string()));
    }

    let inbox = inner.inboxes.get_mut(&user)
        .ok_or_else(|| (StatusCode::NOT_FOUND, "Inbox not found".to_string()))?;

    // Drain and clear the user's inbox
    let messages = std::mem::take(inbox);
    Ok(Json(messages))
}

#[tokio::main]
async fn main() {
    println!("Loading moderation model from {} and building circuit...", MODEL_PATH);
    let model = Model::from_json_file(MODEL_PATH)
        .expect("failed to load moderation model (run moderation/train.py first)");
    let circuit = Arc::new(ModerationCircuit::new(model));
    println!("Circuit ready (d = {}). Server acts as a zero-knowledge middlebox.",
             circuit.model.d);

    let inner = Arc::new(Mutex::new(ServerState {
        users: HashSet::new(),
        bundles: HashMap::new(),
        inboxes: HashMap::new(),
    }));
    let state = AppState { inner, circuit };

    let app = Router::new()
        .route("/register", post(register))
        .route("/bundles/:user", post(publish_bundle).get(fetch_bundle))
        .route("/inbox/:user", post(enqueue_message).get(drain_inbox))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind("127.0.0.1:3000")
        .await
        .unwrap();
    println!("Server running on http://127.0.0.1:3000");
    axum::serve(listener, app).await.unwrap();
}
