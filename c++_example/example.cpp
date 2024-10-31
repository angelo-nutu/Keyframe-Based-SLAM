#include <iostream>
#include <vector>
#include <zmq.hpp>

int main() {
    zmq::context_t context(1);
    zmq::socket_t socket(context, ZMQ_SUB);
    socket.connect("tcp://localhost:5555");  
    socket.setsockopt(ZMQ_SUBSCRIBE, "", 0); // Subscribe to all messages

    while (true) {
        zmq::message_t message;
        socket.recv(&message);

        // Calculate number of floats in the message
        size_t num_floats = message.size() / sizeof(float);
        std::vector<float> positions(num_floats);
        
        // Copy the raw data into the vector
        std::memcpy(positions.data(), message.data(), message.size());

        // Process the positions as (x, y, z) tuples
        for (size_t i = 0; i < num_floats; i += 3) {
            float x = positions[i];
            float y = positions[i + 1];
            float z = positions[i + 2];
            std::cout << "Cone Position: [" << x << ", " << y << ", " << z << "]\n";
        }
    }

    return 0;
}
