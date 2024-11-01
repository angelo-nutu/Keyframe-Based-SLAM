#include <iostream>
#include <vector>
#include <zmq.hpp>
#include <cstring>

int main() {
    zmq::context_t context(1);
    zmq::socket_t socket(context, ZMQ_SUB);
    socket.connect("tcp://localhost:5555");
    socket.setsockopt(ZMQ_SUBSCRIBE, "", 0); // Subscribe to all messages

    while (true) {
        zmq::message_t message;
        socket.recv(&message);

        size_t total_size = message.size();
        if (total_size < sizeof(float) * 3 + sizeof(int)) {
            continue; 
        }

        // Calculate the number of points
        size_t num_points = total_size / (sizeof(float) * 3 + sizeof(int));

        std::vector<float> positions; 
        std::vector<int> class_ids; 

        // Extract points and class IDs
        for (size_t i = 0; i < num_points; ++i) {
            float x, y, z;
            int class_id;

            // I don't get it pointers are weird, everything explodes otherwise
            char* data = static_cast<char*>(message.data());

            // :)
            std::memcpy(&x, data + i * (sizeof(float) * 3 + sizeof(int)), sizeof(float));
            std::memcpy(&y, data + i * (sizeof(float) * 3 + sizeof(int)) + sizeof(float), sizeof(float));
            std::memcpy(&z, data + i * (sizeof(float) * 3 + sizeof(int)) + sizeof(float) * 2, sizeof(float));

            // :D
            std::memcpy(&class_id, data + i * (sizeof(float) * 3 + sizeof(int)) + sizeof(float) * 3, sizeof(int));

            // Store the values
            positions.push_back(x);
            positions.push_back(y);
            positions.push_back(z);
            class_ids.push_back(class_id);
        }

        // Print each (x, y, z) with the corresponding class ID!!!!!!!!!!!!!!
        for (size_t i = 0; i < positions.size() / 3; ++i) {
            float x = positions[i * 3];
            float y = positions[i * 3 + 1];
            float z = positions[i * 3 + 2];
            int class_id = class_ids[i];
            std::cout << "Point Position: [" << x << ", " << y << ", " << z << "], Class ID: " << class_id << "\n";
        }
    }

    return 0;
}
