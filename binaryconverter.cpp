#include <iostream>
#include <string>
#include <cmath>
using namespace std;

// Convert denary to binary (returns string)
string denaryToBinary(int num) {
    if (num == 0) return "0";
    string binary = "";
    while (num > 0) {
        binary = to_string(num % 2) + binary;
        num /= 2;
    }
    return binary;
}

// Convert binary string to denary
int binaryToDenary(const string& bin) {
    int result = 0;
    for (char c : bin) {
        if (c != '0' && c != '1') {
            cout << "Invalid binary number!" << endl;
            return -1;
        }
        result = result * 2 + (c - '0');
    }
    return result;
}

int main() {
    int choice;
    cout << "Choose conversion mode:" << endl;
    cout << "1. Denary to Binary" << endl;
    cout << "2. Binary to Denary" << endl;
    cout << "Enter choice (1 or 2): ";
    cin >> choice;

    if (choice == 1) {
        int denary;
        cout << "Enter a denary (decimal) number: ";
        cin >> denary;
        cout << "Binary: " << denaryToBinary(denary) << endl;
    } else if (choice == 2) {
        string binary;
        cout << "Enter a binary number: ";
        cin >> binary;
        int denary = binaryToDenary(binary);
        if (denary != -1)
            cout << "Denary: " << denary << endl;
    } else {
        cout << "Invalid choice." << endl;
    }

    return 0;
}
