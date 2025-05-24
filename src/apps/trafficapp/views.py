from django.shortcuts import render

# Create your views here.
from django.views.generic import TemplateView
from django.conf import settings
from django.http import JsonResponse
from django.utils.timezone import now

from .models import VehicleDetection, TrafficCount

def index(request):
    """
    Index view that redirects to the dashboard.
    """
    context = {
        'media_url': settings.MEDIA_URL,
    }
    # Redirect to the dashboard view
    return render(request, 'traffic/index.html', context)

class DashboardView(TemplateView):
    """
    Main dashboard view that displays the traffic monitoring interface.
    """
    template_name = 'traffic/dashboard.html'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        
        # Get latest traffic statistics
        context['total_count'] = TrafficCount.objects.latest('timestamp').total_count if TrafficCount.objects.exists() else 0
        
        # Get vehicle counts by type
        context['vehicle_counts'] = VehicleDetection.objects.all()
        
        # Add any additional context needed for the dashboard
        context['last_updated'] = now().strftime("%Y-%m-%d %H:%M:%S")
        
        return context
    
    
def traffic_data_api(request):
    """
    JSON API endpoint that returns the latest traffic data.
    """
    # Get latest traffic statistics
    total_count = TrafficCount.objects.latest('timestamp').total_count if TrafficCount.objects.exists() else 0
    
    # Get counts by vehicle type
    vehicle_counts = {}
    for detection in VehicleDetection.objects.all():
        vehicle_counts[detection.vehicle_type] = detection.count
    
    # Build response
    response_data = {
        'total_count': total_count,
        'vehicle_counts': vehicle_counts,
        'last_updated': now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    return JsonResponse(response_data)

    
def reset_counters(request):
    """
    Administrative function to reset all traffic counters.
    """
    if request.method == 'POST':
        # Reset all vehicle detection counts
        VehicleDetection.objects.all().update(count=0)
            
        # Create a new traffic count record with zero
        TrafficCount.objects.create(total_count=0)
            
        return JsonResponse({'status': 'success', 'message': 'Counters reset successfully'})
        
    return JsonResponse({'status': 'error', 'message': 'Method not allowed'}, status=405)
