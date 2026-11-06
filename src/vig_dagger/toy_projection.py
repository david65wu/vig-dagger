"""Camera projection using calibrated rigid transforms and recorded TCP poses."""
import numpy as np


def rotation(q):
    w,x,y,z=np.asarray(q,dtype=float)/np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])


def camera_pose(calibration,state,camera):
    c=calibration[camera]
    if camera=='third_person':return np.array(c['position_world']),np.array(c['rotation_ros_world'])
    root_rotation=np.array(calibration['root_rotation']);root_position=np.array(calibration['root_position'])
    tcp_rotation=root_rotation@rotation(state[3:7]);tcp_position=root_position+root_rotation@state[:3]
    return tcp_position+tcp_rotation@c['tcp_relative_position'],tcp_rotation@c['tcp_relative_rotation']


def project(calibration,state,camera,points_base):
    position,orientation=camera_pose(calibration,state,camera)
    points_world=np.asarray(points_base)@np.array(calibration['root_rotation']).T+calibration['root_position']
    camera_points=(points_world-position)@orientation
    pixels=camera_points@np.array(calibration[camera]['intrinsics']).T
    depth=camera_points[:,2]
    return pixels[:,:2]/np.where(np.abs(depth)>1e-9,depth,1e-9)[:,None],depth
