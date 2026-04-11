from tracker.ball_detector import BallDetector

bd = BallDetector()
bd.tune_hsv(0)

# import pyrealsense2 as rs                                                                                                                                                                                        
# ctx = rs.context()                                                                                                                                                                                               
# print(f"Devices found: {len(ctx.query_devices())}")                                                                                                                                                              
# for dev in ctx.query_devices():                                                                                                                                                                                  
#     print(f"  {dev.get_info(rs.camera_info.name)}")