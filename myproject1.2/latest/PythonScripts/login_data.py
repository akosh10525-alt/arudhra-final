import json

def save_login(username, password):
    if username and password:
        data = {"username": username, "password": password}
        with open("JsonFiles/login.json", "w") as f:
            json.dump(data, f, indent=4)
        print("Login details saved.")

def save_signup_af(service_number, rank, name, password):
    if service_number and rank and name and password:
        data = {
            "service_number": service_number,
            "rank": rank,
            "name": name,
            "password": password
        }
        with open("JsonFiles/signup_af.json", "w") as f:
            json.dump(data, f, indent=4)
        print("Air Force Personnel signup details saved.")

def save_signup_other(name, phone, email, password):
    if name and phone and email and password:
        data = {
            "name": name,
            "phone": phone,
            "email": email,
            "password": password
        }
        with open("JsonFiles/signup_other.json", "w") as f:
            json.dump(data, f, indent=4)
        print("Other signup details saved.")


# Example usage
# if __name__ == "__main__":
#     save_login("test_user", "password123")
#     save_signup_af("12345", "Flight Lt", "John Doe", "securepass")
#     save_signup_other("Jane Smith", "9876543210", "jane@example.com", "mypassword")
