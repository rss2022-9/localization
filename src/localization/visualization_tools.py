from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker

class VisualizationTools:

    @staticmethod
    def plot_cloud(x, y, publisher, color = (1., 0., 0.), frame = "/map"):
        
        # Construct cloud
        point_cloud = Marker()
        point_cloud.type = Marker.POINTS
        point_cloud.header.frame_id = frame

        # Set the size and color
        point_cloud.scale.x = 0.1
        point_cloud.scale.y = 0.1
        point_cloud.color.a = 1.
        point_cloud.color.r = color[0]
        point_cloud.color.g = color[1]
        point_cloud.color.g = color[2]

        # Fill cloud
        for xi, yi in zip(x, y):
            p = Point()
            p.x = xi
            p.y = yi
            point_cloud.points.append(p)

        # Publish the cloud
        publisher.publish(point_cloud)