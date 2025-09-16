from flask import Flask, request, jsonify, session, redirect, url_for, send_from_directory, abort
from PythonScripts import login_backend as backend
from werkzeug.utils import secure_filename
from datetime import datetime
import os, json

app = Flask(__name__)
app.secret_key = "supersecretkey"

# Root paths
ROOT = app.root_path
HTML_DIR = os.path.join(ROOT, "html")
IMAGES_DIR = os.path.join(ROOT, "Images")
PRECIS_DIR = os.path.join(ROOT, "PrecisDoc")
QUERY_DIR = os.path.join(ROOT, "Query")
JSON_DIR = os.path.join(ROOT, "JsonFiles")

# Ensure folders exist
os.makedirs(HTML_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(PRECIS_DIR, exist_ok=True)
os.makedirs(QUERY_DIR, exist_ok=True)
os.makedirs(JSON_DIR, exist_ok=True)

# File paths
CREDENTIALS_FILE = os.path.join(ROOT, "credentials.json")
AF_SIGNUP_FILE = os.path.join(JSON_DIR, "signup_af.json")
OTHER_SIGNUP_FILE = os.path.join(JSON_DIR, "signup_other.json")

# ---------------- Utility ----------------
def load_json(path):
    if os.path.exists(path):
        with open(path, "r") as f:
            try:
                return json.load(f)
            except:
                return []
    return []

def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


# ---------------- Serve Static Files ----------------
@app.route("/Images/<path:filename>")
def images(filename):
    return send_from_directory(IMAGES_DIR, filename)

@app.route("/PrecisDoc/<path:filename>")
def precis_doc(filename):
    safe = os.path.normpath(filename)
    if safe.startswith("..") or os.path.isabs(safe):
        abort(400)
    return send_from_directory(PRECIS_DIR, safe)

def serve_html(filename):
    return send_from_directory(HTML_DIR, filename)


# ---------------- Page Routes ----------------
@app.route("/")
def index():
    return serve_html("login_page.html")

@app.route("/dashboard")
def dashboard_page():
    return serve_html("dashboard.html")

@app.route("/precis_documents")
def precis_documents_page():
    return serve_html("precis_document.html")

@app.route("/query")
def query_page():
    return serve_html("query.html")

@app.route("/manage_queries")
def manage_queries_page():
    return serve_html("manage_queries.html")

@app.route("/manage_users")
def manage_users_page():
    return serve_html("manage_users.html")


# ---------------- Auth API ----------------
@app.route("/login", methods=["POST"])
def login():
    data = request.get_json() or {}
    username = data.get("username", "")
    password = data.get("password", "")

    result = backend.login(username, password)
    if result.get("status") == "success":
        session["user"] = {
            "username": username,
            "name": result.get("name"),
            "rank": result.get("rank"),
            "authority": result.get("authority"),
        }
    return jsonify(result)

@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("index"))

@app.route("/api/user")
def api_user():
    user = session.get("user")
    if not user:
        return jsonify({"status": "error", "message": "Not logged in"}), 401
    return jsonify({"status": "success", "user": user})


# ---------------- Signup APIs ----------------
@app.route("/signup_af", methods=["POST"])
def signup_af():
    data = request.get_json() or {}
    service_number = data.get("service_number", "")
    rank = data.get("rank", "")
    name = data.get("name", "")
    password = data.get("password", "")
    authority = data.get("authority", None)

    result = backend.save_signup_af(service_number, rank, name, password, authority)
    return jsonify(result)

@app.route("/signup_other", methods=["POST"])
def signup_other():
    data = request.get_json() or {}
    phone_number = data.get("phone_number", "")
    name = data.get("name", "")
    password = data.get("password", "")
    authority = data.get("authority", None)

    result = backend.save_signup_other(phone_number, name, password, authority)
    return jsonify(result)


# ---------------- Precis Documents API ----------------
@app.route("/api/documents")
def api_documents():
    docs = [f for f in os.listdir(PRECIS_DIR) if f.lower().endswith(".pdf")]
    docs.sort()
    return jsonify({"status": "success", "documents": docs})

@app.route("/upload_precis", methods=["POST"])
def upload_precis():
    user = session.get("user")
    if not user:
        return jsonify({"status": "error", "message": "Not logged in"}), 401
    if user.get("authority") not in [0, 1]:
        return jsonify({"status": "error", "message": "Permission denied"}), 403

    if "pdf_file" not in request.files:
        return jsonify({"status": "error", "message": "No file uploaded"}), 400

    file = request.files["pdf_file"]
    filename = secure_filename(file.filename or "")
    if not filename.lower().endswith(".pdf"):
        return jsonify({"status": "error", "message": "Only PDF files allowed"}), 400

    save_path = os.path.join(PRECIS_DIR, filename)
    file.save(save_path)
    return jsonify({"status": "success", "message": "File uploaded successfully", "filename": filename})


# ---------------- Query APIs ----------------
@app.route("/api/query", methods=["POST"])
def api_query():
    user = session.get("user")
    if not user:
        return jsonify({"status": "error", "message": "Not logged in"}), 401

    data = request.get_json() or {}
    query_text = data.get("feedback", "").strip()
    if not query_text:
        return jsonify({"status": "error", "message": "Query cannot be empty"})

    query_file = os.path.join(QUERY_DIR, "queries.json")
    existing = load_json(query_file)

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = {
        "name": user.get("name"),
        "rank": user.get("rank"),
        "authority": user.get("authority"),
        "query": query_text,
        "datetime": now,
        "status": "Unresolved"
    }
    existing.append(entry)
    save_json(query_file, existing)

    return jsonify({"status": "success", "message": "Query submitted successfully!"})


@app.route("/api/manage_queries")
def api_manage_queries():
    user = session.get("user")
    if not user or user.get("authority") not in [0, 1]:
        return jsonify({"status": "error", "message": "Access denied"}), 403

    query_file = os.path.join(QUERY_DIR, "queries.json")
    queries = load_json(query_file)
    return jsonify({"status": "success", "queries": queries})


@app.route("/api/update_query", methods=["POST"])
def api_update_query():
    user = session.get("user")
    if not user or user.get("authority") not in [0, 1]:
        return jsonify({"status": "error", "message": "Access denied"}), 403

    data = request.get_json() or {}
    index = data.get("index")
    status = data.get("status")

    query_file = os.path.join(QUERY_DIR, "queries.json")
    queries = load_json(query_file)

    if index is None or index < 0 or index >= len(queries):
        return jsonify({"status": "error", "message": "Invalid query index"})

    queries[index]["status"] = status
    save_json(query_file, queries)

    return jsonify({"status": "success", "message": f"Query marked as {status}"})


# ---------------- User Creation & Management APIs ----------------
@app.route("/api/create_user", methods=["POST"])
def api_create_user():
    user = session.get("user")
    if not user or user.get("authority") not in [0, 1]:
        return jsonify({"status": "error", "message": "Access denied"}), 403

    data = request.get_json() or {}
    creds = load_json(CREDENTIALS_FILE)

    if "service_number" in data:
        new_user = {
            "username": data["service_number"],
            "password": data["password"],
            "authority": data.get("authority", 0),
            "rank": data.get("rank", ""),
            "name": data.get("name", "")
        }
    else:
        new_user = {
            "username": data["phone_number"],
            "password": data["password"],
            "authority": data.get("authority", 2),
            "rank": "N/A",
            "name": data.get("name", "")
        }

    creds.append(new_user)
    save_json(CREDENTIALS_FILE, creds)
    return jsonify({"status": "success", "message": "User created successfully"})


@app.route("/api/user_requests")
def api_user_requests():
    user = session.get("user")
    if not user or user.get("authority") not in [0, 1]:
        return jsonify({"status": "error", "message": "Access denied"}), 403

    af_requests = load_json(AF_SIGNUP_FILE)
    oth_requests = load_json(OTHER_SIGNUP_FILE)
    return jsonify({"status": "success", "af": af_requests, "other": oth_requests})


@app.route("/api/handle_request", methods=["POST"])
def api_handle_request():
    user = session.get("user")
    if not user or user.get("authority") not in [0, 1]:
        return jsonify({"status": "error", "message": "Access denied"}), 403

    data = request.get_json() or {}
    req_type = data.get("type")
    index = data.get("index")
    approve = data.get("approve")

    creds = load_json(CREDENTIALS_FILE)
    if req_type == "af":
        requests = load_json(AF_SIGNUP_FILE)
    else:
        requests = load_json(OTHER_SIGNUP_FILE)

    if index is None or index < 0 or index >= len(requests):
        return jsonify({"status": "error", "message": "Invalid request index"})

    req_user = requests.pop(index)
    if approve:
        creds.append(req_user)
        save_json(CREDENTIALS_FILE, creds)
        message = "Request approved and user added"
    else:
        message = "Request rejected"

    if req_type == "af":
        save_json(AF_SIGNUP_FILE, requests)
    else:
        save_json(OTHER_SIGNUP_FILE, requests)

    return jsonify({"status": "success", "message": message})


@app.route("/api/all_users")
def api_all_users():
    user = session.get("user")
    if not user or user.get("authority") not in [0, 1]:
        return jsonify({"status": "error", "message": "Access denied"}), 403

    users = load_json(CREDENTIALS_FILE)
    return jsonify({"status": "success", "users": users})


@app.route("/api/delete_user", methods=["POST"])
def api_delete_user():
    user = session.get("user")
    if not user or user.get("authority") not in [0, 1]:
        return jsonify({"status": "error", "message": "Access denied"}), 403

    data = request.get_json() or {}
    index = data.get("index")
    users = load_json(CREDENTIALS_FILE)

    if index is None or index < 0 or index >= len(users):
        return jsonify({"status": "error", "message": "Invalid index"})

    users.pop(index)
    save_json(CREDENTIALS_FILE, users)
    return jsonify({"status": "success", "message": "User deleted"})


# ---------------- Run ----------------
if __name__ == "__main__":
    app.run(debug=True)
