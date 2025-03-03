###### Create by Bui Quang Minh and friends on 28/11/2024 ######

import os
import sys
import cv2
from Prohibited_Zone_Service import process_video as prohibited_zone_service
from Crowd_Detection_Service2 import process_video as crowd_detection_service
# from People_Counting_Service import process_video as people_counting_service
from People_Counting_Service import process_video as people_counting_service

def select_model():
    print("\n===== Model Selection =====")
    print("\n Caution !!! Bigger model will have better detect quality, but reduce of performance")
    print("1. Model Nano")
    print("2. Model Small")
    print("3. Model Medium")
    print("4. Model Large")
    print("5. Model Extreme")
    print("===========================")

    model_choice = None
    while model_choice not in [1, 2, 3, 4, 5]:
        try:
            raw_input = input("Choose a model by entering 1, 2, 3, 4, or 5: ").strip()
            if not raw_input:
                print("No input detected. Please enter a valid choice.")
                continue
            model_choice = int(raw_input)
            if model_choice not in [1, 2, 3, 4, 5]:
                print("Invalid choice! Please enter a number (1, 2, 3, 4, or 5).")
        except ValueError:
            print("Invalid input! Please enter a number (1, 2, 3, 4, or 5).")

    model_paths = {
        1: "model\yolov8n_training.pt",
        2: "model\yolov8s_training.pt",
        3: "model\yolov8m_training.pt",
        4: "model\yolov8l_training.pt",
        5: "model\yolov8x_training.pt",
    }
    return model_paths[model_choice]

def main():
    print("===== Video Processing Services =====")
    print("1. Prohibited Zone Service")
    print("2. Crowd Detection Service")
    print("3. People Counting Service")
    print("=====================================")

    choice = None
    while choice not in [1, 2, 3]:
        try:
            raw_input = input("Choose a service by entering 1, 2, or 3: ").strip()
            if not raw_input:  
                print("No input detected. Please enter a valid choice.")
                continue
            choice = int(raw_input)
            if choice not in [1, 2, 3]:
                print("Invalid choice! Please enter a number (1, 2, or 3).")
        except ValueError:
            print("Invalid input! Please enter a number (1, 2, or 3).")

    print(f"Service selected: {choice}")

    input_path = input("Enter the path of the input video(or enter 0 to use web cam): ").strip()
    if input_path == "0":
        input_path = 0
    elif not os.path.isfile(input_path):
        print(f"Error: The file '{input_path}' does not exist.")
        return

    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"output_service_{choice}.mp4")
    print(f"Output path: {output_path}")

    model_path = select_model()
    print(f"Model selected: {model_path}")

    if choice == 1:
        # points = []
        print("Running Prohibited Zone Service...")
        prohibited_zone_service(input_path, output_path, model_path)
    elif choice == 2:
        print("Running Crowd Detection Service...")
        crowd_detection_service(input_path, output_path, model_path)
    elif choice == 3:
        print("Running People Counting Service...")
        people_counting_service(input_path, output_path, model_path)

    print(f"Processing completed. Output saved at: {output_path}")

if __name__ == "__main__":
    main()