import argparse
import cv2
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Test YOLO model on webcam")
    parser.add_argument("checkpoint", type=str, help="Path to model checkpoint (.pt file)")
    parser.add_argument("--webcam", type=int, default=0, help="Webcam index (default: 0)")
    args = parser.parse_args()

    model = YOLO(args.checkpoint, task="segment")
    cap = cv2.VideoCapture(args.webcam)

    if not cap.isOpened():
        print(f"Error: Could not open webcam {args.webcam}")
        return

    print(f"Model loaded: {args.checkpoint}")
    print(f"Running model on webcam {args.webcam}. Press 'q' to quit.")
    print("If no window appears, click on the terminal and try again.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break

        results = model(frame, verbose=False)

        # Debug: print detection count
        num_detections = len(results[0].boxes) if results[0].boxes is not None else 0
        print(f"Frame processed - {num_detections} detections")

        annotated = results[0].plot()
        cv2.imshow("YOLO Test", annotated)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
