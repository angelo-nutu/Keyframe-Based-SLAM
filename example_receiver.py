import zmq
import msgpack

def main():
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.connect("tcp://localhost:5555")
    socket.setsockopt_string(zmq.SUBSCRIBE, '')

    while True:
        msg = socket.recv()
        point = msgpack.unpackb(msg)
        
        print(f"Received: {point}")

if __name__ == "__main__":
    main()
