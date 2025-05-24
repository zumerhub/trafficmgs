from django.db import models

# Create your models here.
class VehicleDetection(models.Model):
    """
    Model to store vehicle type detection results with timestamps
    """
    VEHICLE_CHOICES = [
        ('car', 'Car'),
        ('bus', 'Bus'),
        ('truck', 'Truck'),
        ('mini-bus', 'Mini-bus'),
        ('motorcycle', 'Motorcycle'), 
    ]
    
    # vehicle_type = models.CharField(max_length=50, choices=VECHICLE_CHOICES, unique=True)
    # count = models.IntegerField(default=0)
    vehicle_type = models.CharField(max_length=50, choices=VEHICLE_CHOICES)
    count = models.IntegerField(default=0)
    timestamp = models.DateTimeField(auto_now_add=True)


    # def __str__(self):
    #     return f"{self.vehicle_type}, Count: {self.count}, Timestamp: {self.timestamp.strftime('%Y-%m-%d %H:%M')}"
    
    def __str__(self):
        return f"{self.vehicle_type} x{self.count} at {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}"

    # Add method to update count of existing detections
    @classmethod
    def update_or_create(cls, vehicle_type):
        obj, created = cls.objects.get_or_create(vehicle_type=vehicle_type)
        if created:
            obj.count = 1  # Initialize count if the record is new
        else:
            obj.count += 1  # Increment the count if the record exists
        obj.save()
        return obj
    
    def log_detection(cls, vehicle_type, count=1):
        """
        Log a new detection instance.
        """
        return cls.objects.create(vehicle_type=vehicle_type, count=count)

class VehicleDetectionLog(models.Model):
    """
    Model to store individual vehicle detection events
    """
    vehicle_type = models.CharField(max_length=50)
    count = models.IntegerField(default=1)
    timestamp = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"{self.vehicle_type} detected at {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}"


class TrafficCount(models.Model):
    """
    Model to store traffic count results with timestamps
    """
    timestamp = models.DateTimeField(auto_now_add=True)
    total_count = models.IntegerField()

    def __str__(self):
        return f"{self.timestamp} - Count: {self.total_count}"
    