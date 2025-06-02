

import json
import os
import base64
import cv2
import logging
import asyncio
import numpy as np
from django.utils.timezone import now
from ultralytics import YOLO
from django.utils import timezone
from channels.generic.websocket import AsyncWebsocketConsumer
from channels.db import database_sync_to_async
from .sort import *

import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'trafficsite.settings')
django.setup()

from apps.trafficapp.models import VehicleDetection

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class TrafficConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        logger.info("WebSocket connection attempt")
        await self.accept()
        logger.info("WebSocket Connected!!!!!")
        self.running = True
        
        # Start the video processing task
        asyncio.create_task(self.send_traffic_update())

    async def disconnect(self, close_code):
        logger.info(f"WebSocket disconnected with code: {close_code}")
        self.running = False
    
    async def receive(self, text_data):
        """Handle incoming WebSocket messages."""
        try:
            data = json.loads(text_data)
            logger.info(f"Received data: {data}")

            if data.get("action") == "detect_traffic":
                await self.send(text_data=json.dumps({
                    "message": "Detection already in progress"
                }))
        except Exception as e:
            logger.error(f"Error processing received message: {e}")
    
    @database_sync_to_async
    def update_vehicle_detection(self, vehicle_type):
        from .models import VehicleDetection
        logger.info(f"Updating vehicle detection for type: {vehicle_type}")

        try:
            # Use the class method from your model
            obj = VehicleDetection.update_or_create(vehicle_type)
            logger.info(f"Updated/Created vehicle detection: {obj}")
            return obj
        except Exception as e:
            logger.error(f"Error updating vehicle detection: {e}")
            return None
    
    @database_sync_to_async
    def log_vehicle_detection(self, vehicle_type):
        from .models import VehicleDetectionLog
        
        try:
            VehicleDetectionLog.objects.create(
                vehicle_type=vehicle_type,
                count=1,
                timestamp=timezone.now()
            )
        except Exception as e:
            logger.error(f"Error logging vehicle detection: {e}")
    
    @database_sync_to_async
    def update_traffic_count(self, count):
        from .models import TrafficCount
        
        try:
            traffic_count = TrafficCount.objects.create(
                total_count=count,
                timestamp=timezone.now()
            )
            return traffic_count
        except Exception as e:
            logger.error(f"Error updating traffic count: {e}")
            return None

    async def send_traffic_update(self):
        """Process video and send updates via WebSocket."""
        try:
            # Load YOLO model
            MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "/home/zumerhub/codebase/trafficmgs/ml/model/my_model11.pt")
            self.model = YOLO(MODEL_PATH)
            logger.info("YOLO model loaded successfully!")
            
            # Video source
            # VIDEO_SOURCE = os.getenv("VIDEO_SOURCE", "/home/zumerhub/codebase/trafficmgs/src/apps/trafficapp/static/video/cars.mp4") #"apps/trafficapp/static/video/mile2Bridge1.mp4")
            VIDEO_SOURCE = os.getenv("VIDEO_SOURCE", "/home/zumerhub/codebase/trafficmgs/src/apps/trafficapp/static/video/mile2Bridge1.mp4") #"apps/trafficapp/static/video/mile2Bridge1.mp4")
            # VIDEO_SOURCE = os.getenv("WEB_CAM", 2)
            
            # Class names for detection
            class_names = ['bus', 'car', 'mini-bus', 'motorcycle', 'truck']
            
            # Initialize tracker
            tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
            
            # Region of interest (counting line)
            limits = (245, 216, 897, 216)  # mile2Bridge1
            
            # Total count of vehicles and tracking data
            total_count = []
            tracked_vehicles = {}  # Store vehicle type for each tracked ID
            
            # Initialize video capture
            cap = cv2.VideoCapture(VIDEO_SOURCE)
            if not cap.isOpened():
                logger.error(f"Failed to open video source: {VIDEO_SOURCE}")
                await self.send(text_data=json.dumps({
                    "error": f"Failed to open video source: {VIDEO_SOURCE}"
                }))
                return
            
            logger.info(f"Successfully opened video source: {VIDEO_SOURCE}")
            
            # Process frames while the WebSocket is connected
            while self.running:
                success, img = cap.read()
                if not success:
                    logger.error("Failed to read frame")
                    await asyncio.sleep(1)
                    continue
                
                # Resize image for processing
                resized_img = cv2.resize(img, (960, 540))
                
                # Perform detection
                results = self.model(resized_img, stream=True)
                
                # Initialize detections array and detection info
                detections = np.empty((0, 5))
                detection_info = []  # Store detection info with class names
                
                # Process results
                for r in results:
                    for box in r.boxes:
                        # Get box coordinates
                        x1, y1, x2, y2 = box.xyxy[0]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        
                        # Get confidence
                        conf = float(box.conf[0])
                        
                        # Get class
                        cls = int(box.cls[0])
                        if cls >= len(class_names):
                            logging.warning(f"Skipping unknown class index: {cls}")
                            continue
                        
                        current_class = class_names[cls]
                        
                        # Filter detections
                        if current_class in ['mini-bus', 'bus', 'motorcycle', 'truck', 'car'] and conf > 0.2:
                            # Add to detections array
                            current_array = np.array([x1, y1, x2, y2, conf])
                            detections = np.vstack((detections, current_array)) if detections.size else np.array([current_array])
                            
                            # Store detection info
                            detection_info.append({
                                'bbox': [x1, y1, x2, y2],
                                'class': current_class,
                                'conf': conf
                            })
                
                # Update tracker with new detections
                results_tracker = tracker.update(detections)
                
                # Match tracked objects with detections to get vehicle types
                for i, result in enumerate(results_tracker):
                    x1, y1, x2, y2, track_id = result
                    x1, y1, x2, y2, track_id = int(x1), int(y1), int(x2), int(y2), int(track_id)
                    
                    # Find the best matching detection for this tracked object
                    best_match = None
                    best_overlap = 0
                    
                    for detection in detection_info:
                        dx1, dy1, dx2, dy2 = detection['bbox']
                        
                        # Calculate overlap/intersection
                        overlap_x1 = max(x1, dx1)
                        overlap_y1 = max(y1, dy1)
                        overlap_x2 = min(x2, dx2)
                        overlap_y2 = min(y2, dy2)
                        
                        if overlap_x1 < overlap_x2 and overlap_y1 < overlap_y2:
                            overlap_area = (overlap_x2 - overlap_x1) * (overlap_y2 - overlap_y1)
                            tracker_area = (x2 - x1) * (y2 - y1)
                            detection_area = (dx2 - dx1) * (dy2 - dy1)
                            
                            # Calculate IoU (Intersection over Union)
                            union_area = tracker_area + detection_area - overlap_area
                            iou = overlap_area / union_area if union_area > 0 else 0
                            
                            if iou > best_overlap:
                                best_overlap = iou
                                best_match = detection
                    
                    # Store vehicle type for this tracked ID
                    if best_match and best_overlap > 0.3:  # Minimum IoU threshold
                        tracked_vehicles[track_id] = best_match['class']
                
                # Draw region of interest line
                cv2.line(resized_img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), thickness=5)
                
                # Process each tracked object for counting
                for result in results_tracker:
                    x1, y1, x2, y2, track_id = result
                    x1, y1, x2, y2, track_id = int(x1), int(y1), int(x2), int(y2), int(track_id)
                    
                    # Get vehicle type for this tracked object
                    vehicle_type = tracked_vehicles.get(track_id, 'unknown')
                    
                    # Draw bounding box
                    cv2.rectangle(resized_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Calculate center point
                    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                    
                    # Draw center point
                    cv2.circle(resized_img, (cx, cy), 5, (0, 0, 255), cv2.FILLED)
                    
                    # Check if vehicle crosses the counting line
                    if limits[0] < cx < limits[2] and limits[1] - 15 < cy < limits[1] + 15:
                        if track_id not in total_count:
                            total_count.append(track_id)
                            
                            logger.info(f"Vehicle {track_id} of type '{vehicle_type}' crossed the line")
                            
                            # Update database (async) - only if we have a valid vehicle type
                            if vehicle_type != 'unknown':
                                await self.update_vehicle_detection(vehicle_type)
                                await self.log_vehicle_detection(vehicle_type)
                                await self.update_traffic_count(len(total_count))
                            
                            # Draw vehicle type and ID
                            cv2.putText(resized_img, f"{vehicle_type} {track_id}", (x1, y1 - 10), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                            
                            # Convert frame to JPEG for streaming
                            _, buffer = cv2.imencode('.jpg', resized_img)
                            stream_video = base64.b64encode(buffer).decode('utf-8')
                            
                            # Send update via WebSocket
                            message = {
                                "vehicle_type": vehicle_type,
                                "count": len(total_count),
                                "timestamp": timezone.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "image": stream_video,
                                "track_id": track_id
                            }
                            
                            await self.send(text_data=json.dumps(message))
                    
                    # Always draw vehicle type and ID for visualization
                    cv2.putText(resized_img, f"{vehicle_type} {track_id}", (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                
                # Add count text to image
                cv2.putText(resized_img, f"Count: {len(total_count)}", (20, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Send periodic updates even when no vehicles cross the line
                if len(total_count) % 5 == 0 or len(total_count) == 0:  # Every 5 vehicles or initially
                    _, buffer = cv2.imencode('.jpg', resized_img)
                    stream_video = base64.b64encode(buffer).decode('utf-8')
                    
                    message = {
                        "vehicle_type": "update",
                        "count": len(total_count),
                        "timestamp": timezone.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "image": stream_video,
                        "total_vehicles": len(tracked_vehicles)
                    }
                    
                    await self.send(text_data=json.dumps(message))
                
                # Small delay to prevent overwhelming the WebSocket
                await asyncio.sleep(0.1)
                    
            # Release resources when done
            cap.release()
            
        except Exception as e:
            logger.error(f"Error in traffic processing: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            # Notify client of error
            if self.running:
                await self.send(text_data=json.dumps({
                    "error": f"Video processing error: {str(e)}"
                }))




# Old file content for src/apps/trafficapp/consumers.py

# import json
# import os
# import base64
# import cv2
# import logging

# import asyncio
# import numpy as np
# from django.utils.timezone import now
# from ultralytics import YOLO
# from django.utils import timezone
# from channels.generic.websocket import AsyncWebsocketConsumer
# from channels.db import database_sync_to_async
# from .sort import *


# import os
# import django

# os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'trafficsite.settings')
# django.setup()

# from apps.trafficapp.models import VehicleDetection

# # Setup logging
# logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO)

# class TrafficConsumer(AsyncWebsocketConsumer):
#     async def connect(self):
#         logger.info("WebSocket connection attempt")
#         await self.accept()
#         logger.info("WebSocket Connected!!!!!")
#         self.running = True
        
#         # Start the video processing task
#         asyncio.create_task(self.send_traffic_update())

#     async def disconnect(self, close_code):
#         logger.info(f"WebSocket disconnected with code: {close_code}")
#         self.running = False
    
#     async def receive(self, text_data):
#         """Handle incoming WebSocket messages."""
#         try:
#             data = json.loads(text_data)
#             logger.info(f"Received data: {data}")

#             if data.get("action") == "detect_traffic":
#                 # The main processing is already happening in send_traffic_update
#                 # This is just for responding to explicit requests
#                 await self.send(text_data=json.dumps({
#                     "message": "Detection already in progress"
#                 }))
#         except Exception as e:
#             logger.error(f"Error processing received message: {e}")
    
#     @database_sync_to_async
#     def update_vehicle_detection(self, vehicle_type):
#         from .models import VehicleDetection
#         logger.info(f"Updating vehicle detection for type: {vehicle_type}")

#         try:
#             obj, created = VehicleDetection.objects.get_or_create(
#                 vehicle_type=vehicle_type,
#                 defaults={'count': 1, 'timestamp': timezone.now()}
#             )
            
#             if not created:
#                 obj.count += 1
#                 obj.timestamp = timezone.now()
#                 obj.save()
#                 logger.info(f"Updated vehicle detection: {obj}")
#             else:
#                 logger.info(f"Created new vehicle detection: {obj}")
                
            
#             return obj
#         except Exception as e:
#             logger.error(f"Error updating vehicle detection: {e}")
#             return None
    
#     @database_sync_to_async
#     def log_vehicle_detection(self, vehicle_type):
#         from .models import VehicleDetectionLog
        
#         try:
#             VehicleDetectionLog.objects.create(
#                 vehicle_type=vehicle_type,
#                 count=1,
#                 timestamp=timezone.now()
#             )
#         except Exception as e:
#             logger.error(f"Error logging vehicle detection: {e}")
    
#     @database_sync_to_async
#     def update_traffic_count(self, count):
#         from .models import TrafficCount
        
#         try:
#             traffic_count = TrafficCount.objects.create(
#                 total_count=count,
#                 timestamp=timezone.now()
#             )
#             return traffic_count
#         except Exception as e:
#             logger.error(f"Error updating traffic count: {e}")
#             return None


#     async def send_traffic_update(self):
#         """Process video and send updates via WebSocket."""
#         try:
#             # Load YOLO model
#             # MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "ml/yolov8n.pt")           
#             # MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "ml/best.pt")

#             MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "/home/zumerhub/codebase/trafficmgs/ml/model/my_model11.pt")
#             # MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "/home/zumerhub/codebase/trafficmgs/ml/model/yolov8n.pt")

            

            
#             self.model = YOLO(MODEL_PATH)
#             logger.info("YOLO model loaded successfully!")
            
#             # Video source - webcam (0) or video file
#             # VIDEO_SOURCE = os.getenv("VIDEO_SOURCE", "apps/trafficapp/static/video/cars.mp4")
#             VIDEO_SOURCE = os.getenv("VIDEO_SOURCE", "apps/trafficapp/static/video/mile2Bridge1.mp4")

#             # VIDEO_SOURCE = os.getenv("WEB_CAM", 2)  # 0 is typically the default webcam
            
#             # Class names for detection
#             class_names = ['bus', 'car', 'mini-bus', 'motorcycle', 'truck']   # first 5 classes from the model
#             # ['bus', 'car', 'mini-bus', 'motorcycle', 'truck']   # # Adjusted for specific vehicle types
#             # class_names = ['motorcycle', 'truck', 'mini-bus', 'bus', 'car'] 

#             # class_names = [
#             #         'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
#             #         'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
#             #         'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
#             #     ]

            
#             # Initialize tracker
#             tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
            
#             # Region of interest
#             # limits = (386, 222, 552, 222) # cars
#             limits = (245, 216, 897, 216) # mile2Bridge1
            
#             # Total count of vehicles
#             total_count = []
            
           
            
#             # Initialize video capture
#             cap = cv2.VideoCapture(VIDEO_SOURCE)
#             if not cap.isOpened():
#                 logger.error(f"Failed to open video source: {VIDEO_SOURCE}")
#                 await self.send(text_data=json.dumps({
#                     "error": f"Failed to open video source: {VIDEO_SOURCE}"
#                 }))
#                 return
            
#             logger.info(f"Successfully opened video source: {VIDEO_SOURCE}")
            
#             # Process frames while the WebSocket is connected
#             while self.running:
#                 success, img = cap.read()
#                 if not success:
#                     logger.error("Failed to read frame")
#                     # For video files, we might want to restart; for webcams, we might want to retry
#                     await asyncio.sleep(1)
#                     continue
                
#                 # Resize image for processing
#                 resized_img = cv2.resize(img, (960, 540))
                
#                 # Perform detection
#                 results = self.model(resized_img, stream=True)
                
#                 # Initialize detections array
#                 detections = np.empty((0, 5))
                
#                 # Process results
#                 for r in results:
#                     for box in r.boxes:
#                         # Get box coordinates
#                         x1, y1, x2, y2 = box.xyxy[0]
#                         x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        
#                         # Get confidence
#                         conf = float(box.conf[0])
                        
#                         # Get class
#                         cls = int(box.cls[0])
#                         if cls >= len(class_names):
#                             logging.warning(f"Skipping unknown class index: {cls}")

#                             continue
                        
#                         currentClass = class_names[cls] if cls < len(class_names) else f"Class-{cls}"
                        
#                         # Filter detections
#                         # if currentClass in ["car", "truck", "bus", "motorcycle"] and conf > 0.2:
#                         if currentClass in ['mini-bus', 'bus', 'motorcycle', 'truck', 'car'] and conf > 0.2:
#                             # Add to detections array
#                             currentArray = np.array([x1, y1, x2, y2, conf])
#                             detections = np.vstack((detections, currentArray)) if detections.size else np.array([currentArray])
                
#                 # Update tracker with new detections
#                 resultsTracker = tracker.update(detections)
                
#                 # Draw region of interest line
#                 cv2.line(resized_img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), thickness=5)
                
#                 # Process each tracked object
#                 for result in resultsTracker:
#                     x1, y1, x2, y2, id = result
#                     x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
#                     # Draw bounding box
#                     cv2.rectangle(resized_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
#                     # Calculate center point
#                     cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                    
#                     # Draw center point
#                     cv2.circle(resized_img, (cx, cy), 5, (0, 0, 255), cv2.FILLED)
                    
#                     # Check if vehicle crosses the counting line
#                     if limits[0] < cx < limits[2] and limits[1] - 6 < cy < limits[1] + 20:
#                         if id not in total_count:
#                             total_count.append(id)
                         
                         
#                             vehicle_type = currentClass   # Default fallback
                          
                      
#                             # For each detection result
#                             for r in results:
#                                 for box in r.boxes:
#                                     x1d, y1d, x2d, y2d = box.xyxy[0]
#                                     x1d, y1d, x2d, y2d = int(x1d), int(y1d), int(x2d), int(y2d)
#                                     cls = int(box.cls[0])
                                    
#                                     # Check if this detection overlaps significantly with our tracked object
#                                     if (abs(x1 - x1d) < 30 and abs(y1 - y1d) < 30 and 
#                                         abs(x2 - x2d) < 30 and abs(y2 - y2d) < 30):
#                                         # Found a match, use this class
#                                         vehicle_type = class_names[cls] if cls < len(class_names) else "unknown"
#                                         break
                            
                            
#                                 # Save vehicle type to database using currentClass
#                                 vehicle_detection = VehicleDetection.objects.filter(vehicle_type=currentClass).first()
#                                 if vehicle_detection:
#                                     vehicle_detection.count += 1
#                                     vehicle_detection.timestamp = now()
#                                     vehicle_detection.save()
#                                 else:
#                                     VehicleDetection.objects.create(vehicle_type=currentClass, count=1, timestamp=now())
                        
                           
#                             # Update database (async)
#                             await self.update_vehicle_detection(vehicle_type)
#                             await self.log_vehicle_detection(vehicle_type)
#                             await self.update_traffic_count(len(total_count))

#                             # Draw vehicle type and ID
#                             cv2.putText(resized_img, f"{vehicle_type} {int(id)}", (x1, y1 - 10), 
#                                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
#                             # Draw count
#                             cv2.putText(resized_img, f"Count: {len(total_count)}", (20, 40), 
#                                         cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                           
#                             # Convert frame to JPEG for streaming

#                             _, buffer = cv2.imencode('.jpg', resized_img)
                
#                             streamVideo = base64.b64encode(buffer).decode('utf-8')
#                             message = {
#                                         "vehicle_type": vehicle_type,
#                                         "count": len(total_count),
#                                         "timestamp": timezone.now().strftime("%Y-%m-%d %H:%M:%S"),
#                                         "image": streamVideo,
#                                     }
                                    
#                             await self.send(text_data=json.dumps(message)) 
#                             await asyncio.sleep(0.01)
                
#                 # Add count text to image
#                 cv2.putText(resized_img, f"Count: {len(total_count)}", (20, 40), 
#                             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                
#             # Release resources when done
#             cap.release()
            
#         except Exception as e:
#             logger.error(f"Error in traffic processing: {e}")
#             import traceback
#             logger.error(traceback.format_exc())
            
#             # Notify client of error
#             if self.running:
#                 await self.send(text_data=json.dumps({
#                     "error": f"Video processing error: {str(e)}"
#                 }))
