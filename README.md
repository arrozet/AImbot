# CV-based aimbot for FPS games 

<p align="center">
  <img src="https://raw.githubusercontent.com/arrozet/AImbot/main/!aux/presentation/images/gif/cs2-big.gif" alt="AI-based Aimbot">
</p>

This project implements a real-time detection and interaction system utilizing BetterCam, YOLO, and a Razer mouse controller. It processes video frames to detect targets and can perform actions such as aiming and shooting to the head.

## Features

- **Real-time detection**: processes video frames in real-time to detect targets using the YOLO model.
- **Head position identification**: identifies head positions within detected targets, utilizing SIFT as a fallback method.
- **Automated interaction**: optionally aims and performs actions (e.g., shooting) at detected targets using a Razer mouse controller.
- **Performance monitoring**: calculates and displays average FPS and inference time per frame.
- **Visualization**: displays bounding boxes, labels, and head markers on the video feed for visual debugging.

## Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/arrozet/AImbot.git
   cd AImbot
   ```

2. **Set up a virtual environment** (optional but recommended):

   ```bash
   python -m venv finally-getting-good-at-games
   finally-getting-good-at-games\Scripts\activate # On Linux: source finally-getting-good-at-games/bin/activate
   ```

3. **Install the required dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure the system**:

   - Ensure that the `config.py` file is properly set up with the correct paths and parameters, including `WEIGHTS_PATH`, `DLL_PATH`, and other configuration settings.

## Usage

1. **Connect the Razer Mouse** (if using automated interaction):

2. **Run the Application**:

   ```bash
   python main.py
   ```

3. **Controls**:

   - Press `Ctrl + Q` to pause and resume the detection.
   - Press `Ctrl + T` to quit the application.

## Configuration

The system can be configured through the `config.py` file. Key parameters include:

- `TARGET_FPS`: sets the target frames per second for the camera.
- `DRAW`: enables or disables visualization of detections.
- `SHOOTING`: enables or disables the shooting functionality.
- `AIMING`: enables or disables the aiming functionality.
- `WEIGHTS_PATH`: path to the YOLO model weights.
- `DLL_PATH`: path to the Razer mouse controller DLL.

## Dependencies

The project was developed using Python 3.12.3.  
All the required libraries and their specific versions can be found in the `requirements.txt` file.

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is not licensed and is intended for educational purposes only.  
The author is not responsible for any misuse of this program. Furthermore, the author disclaims all responsibility for any bans or penalties incurred in games as a result of using this software.

## Acknowledgments

- [BetterCam](https://github.com/RootKit-Org/BetterCam)  
- [YOLO](https://github.com/ultralytics/ultralytics)  
- [SunOne Aimbot](https://github.com/SunOner/sunone_aimbot) - Special thanks to SunOne for the Razer mouse DLL file and the weights used in the model.
- [rzctl](https://github.com/Sadmeme/rzctl) - For providing the Razer mouse controller library used in this project.


## Note about Windows Defender

The dynamic library included in the repository, `rzctl.dll`, may be flagged as a virus by Windows Defender (false positive). This file is sourced from [rzctl](https://github.com/Sadmeme/rzctl), an open-source project. If you are concerned about its content, please review the code yourself.

To temporarily disable Windows Defender and test the code (if you have a Razer mouse), follow these steps:

1. Navigate to **Windows Security > Virus & threat protection settings > Manage settings**.
2. Turn off **Real-time protection**.

Once testing is complete, you can re-enable real-time protection.
