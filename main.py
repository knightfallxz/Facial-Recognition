from face_registration import register_face
from face_training import train_face_model
from face_authentication import authenticate_face

def main():
    while True:
        print("1. Register")
        print("2. Login")
        print("Enter 'q' to quit")
        choice = input("Enter your choice (1 for Register, 2 for Login): ")

        if choice == '1':
            username = input("Enter a username for registration: ")
            register_face(username)
            print(f"User {username} registered successfully. Model is being trained in the background...")
            train_face_model(username)  # Train the model immediately after registration

        elif choice == '2':
            username = input("Enter your username for login: ")  # Capture the username for login
            authenticate_face(username)  # Pass the username to authenticate

        elif choice.lower() == 'q':
            print("Exiting the program.")
            break

        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()

