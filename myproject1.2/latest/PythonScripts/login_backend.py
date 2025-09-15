import json
import re
import os

# File paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

CREDENTIALS_FILE = os.path.join(PROJECT_ROOT, "Credentials", "credentials.json")
AF_SIGNUP_FILE = os.path.join(PROJECT_ROOT, "JsonFiles", "signup_af.json")
OTHER_SIGNUP_FILE = os.path.join(PROJECT_ROOT, "JsonFiles", "signup_other.json")


# ------------------ Validation Functions ------------------

def is_valid_service_number(service_number: str) -> bool:
    return service_number.isdigit() and len(service_number) == 6

def is_valid_phone_number(phone_number: str) -> bool:
    return phone_number.isdigit() and len(phone_number) == 10

def is_valid_password(password: str) -> bool:
    # At least one special character, alphanumeric allowed
    return bool(re.match(r'^(?=.*[!@#$%^&*(),.?":{}|<>])[A-Za-z0-9!@#$%^&*(),.?":{}|<>]+$', password))

def is_alpha_only(text: str) -> bool:
    return text.isalpha()


# ------------------ File Utility ------------------

def load_json(filename):
    if os.path.exists(filename):
        with open(filename, "r") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return []
    return []

def save_json(filename, data):
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)


# ------------------ Signup Functions ------------------

def save_signup_af(service_number, rank, name, password, authority):
    if not is_valid_service_number(service_number):
        return {"status": "error", "message": "Service number must be 6 digits only"}
    if not is_alpha_only(rank):
        return {"status": "error", "message": "Rank must contain only alphabets"}
    if not is_alpha_only(name):
        return {"status": "error", "message": "Name must contain only alphabets"}
    if not is_valid_password(password):
        return {
            "status": "error",
            "message": "Password must be alphanumeric and include at least one special character"
        }
    if authority not in [0, 1, 2]:
        return {"status": "error", "message": "Authority must be 0, 1, or 2"}

    data = load_json(AF_SIGNUP_FILE)
    user = {
        "username": service_number,
        "password": password,
        "authority": authority,
        "rank": rank,
        "name": name
    }
    data.append(user)
    save_json(AF_SIGNUP_FILE, data)
    return {"status": "success", "message": "Air Force signup saved successfully"}


def save_signup_other(phone_number, name, password, authority):
    if not is_valid_phone_number(phone_number):
        return {"status": "error", "message": "Phone number must be 10 digits only"}
    if not is_alpha_only(name):
        return {"status": "error", "message": "Name must contain only alphabets"}
    if not is_valid_password(password):
        return {
            "status": "error",
            "message": "Password must be alphanumeric and include at least one special character"
        }
    if authority not in [0, 1, 2]:
        return {"status": "error", "message": "Authority must be 0, 1, or 2"}

    data = load_json(OTHER_SIGNUP_FILE)
    user = {
        "username": phone_number,
        "password": password,
        "authority": authority,
        "rank": "N/A",
        "name": name
    }
    data.append(user)
    save_json(OTHER_SIGNUP_FILE, data)
    return {"status": "success", "message": "Other signup saved successfully"}


# ------------------ Login Function ------------------

def login(service_number, password):
    if not service_number.isdigit() or len(service_number) not in [6, 10]:
        return {
            "status": "error",
            "message": "Username must be numeric (6-digit Service No. or 10-digit Phone)"
        }

    users = load_json(CREDENTIALS_FILE)
    for user in users:
        if user["username"] == service_number and user["password"] == password:
            return {
                "status": "success",
                "message": f"Welcome {user['name']}",
                "name": user["name"],
                "rank": user["rank"],
                "authority": user["authority"]
            }

    return {"status": "error", "message": "Invalid credentials"}
