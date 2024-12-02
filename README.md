# AImbot

This project uses a YOLOv8 model to automatically aim at targets in FPS games in real time.

## **Prerequisites**

Before starting, make sure you have:
- Python 3.8 or higher installed.
- Pip (included with Python).
- Visual Studio Code (optional but recommended for development).

## **Setting Up the Environment**

To avoid dependency conflicts, it's highly recommended to create a virtual environment (`venv`).

### **Create a Virtual Environment**
1. Open your terminal in the project's root directory.
2. Create the virtual environment:
   ```bash
   python -m venv venv
   ```
3. Activate the virtual environment:
   - **Windows:**
     ```bash
     .\venv\Scripts\activate
     ```
   - **macOS/Linux:**
     ```bash
     source venv/bin/activate
     ```

### **3.2 Install Dependencies**
With the virtual environment activated, install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
If `requirements.txt` is not yet generated, manually install the dependencies:
 ```bash
 pip install ultralytics mss pynput opencv-python
 ```

Then, save them to a `requirements.txt` file:
   ```bash
   pip freeze > requirements.txt
   ```

### **3.3 Configure Visual Studio Code (Optional)**
1. Open your project in Visual Studio Code.
2. Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on macOS) and select `Python: Select Interpreter`.
3. Choose the interpreter corresponding to your virtual environment (`venv`).

## **4. Running the Project**
1. Ensure the virtual environment is activated.
2. Run the main project file:
```bash
   python main.py
```

## **5. Notes**
- To ensure the best performance, consider reducing the capture resolution or enabling GPU acceleration if necessary.
- If you encounter any issues, double-check that the dependencies are installed correctly.

---

## **License**
This project is for educational purposes only. Use it responsibly and ensure compliance with the terms and conditions of any software or game you interact with.


