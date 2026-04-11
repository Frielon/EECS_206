std_msgs/Header header

# Ball detected this frame (false = Kalman prediction only)
bool ball_found

# Number of ArUco markers detected
int32 markers_found

# Position in table frame (mm), origin = table center
float64 x
float64 y

# Velocity in table frame (mm/s)
float64 vx
float64 vy

# Raw pixel coordinates (useful for debugging)
float64 pixel_u
float64 pixel_v
