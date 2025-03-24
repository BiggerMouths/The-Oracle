
import socket
import time

def chat_with_oracle(host='0.0.0.0', port=9999):
    while True:
        try:
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.connect((host, port))

            # First greeting
            message = "hello"
            client_socket.send(message.encode())
            response = client_socket.recv(4096).decode()
            print("\nOracle:", response)
            client_socket.close()

            while True:
                # Get user input
                user_input = input("\nYou: ")
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("\nOracle: Farewell!")
                    return

                # Send request to oracle
                client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                client_socket.connect((host, port))
                client_socket.send(user_input.encode())
                
                response = client_socket.recv(4096).decode()
                print("\nOracle:", response)
                
                client_socket.close()

        except ConnectionRefusedError:
            print("Cannot connect to the Oracle. Make sure the server is running.")
            time.sleep(2)
            return
        except Exception as e:
            print(f"An error occurred: {e}")
            time.sleep(2)
            return

if __name__ == "__main__":
    chat_with_oracle()
