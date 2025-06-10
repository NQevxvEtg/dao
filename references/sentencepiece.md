# Navigate to a directory where you keep source code (e.g., your home directory)
cd ~

# Clone the SentencePiece repository from GitHub
git clone https://github.com/google/sentencepiece.git

# Enter the repository directory
cd sentencepiece

# Create a build directory
mkdir build

# Enter the build directory
cd build

# Configure the build with CMake and install it to /usr/local
cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local
make -j$(nproc)

# Install the library system-wide (may require administrator privileges)
sudo make install

sudo ldconfig -v