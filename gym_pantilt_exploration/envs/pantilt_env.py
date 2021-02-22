import gym
import numpy as np
import octomap
import rosbag
from gym import spaces
from scipy.spatial.transform import Rotation as R

class PanTiltEnv(gym.Env):

    def __init__(self,octomap_path='data/cave.bt',odom_bag_path='data/cave.bag',
            action_step=np.pi/6,camera_fov=(100*np.pi/180,85*np.pi/180),
            camera_res=(352,287),camera_range=6):

        # Ground truth octomap
        self.tree_gt = octomap.OcTree(octomap_path.encode())
        self.octomap_res = self.tree_gt.getResolution()

        # Empty octomap to be built during episode
        self.tree_new = octomap.OcTree(self.octomap_res)

        # Camera parameters
        self.camera_fov = camera_fov
        self.camera_res = camera_res
        self.camera_range = camera_range

        # Load odometry into an array
        bag = rosbag.Bag(odom_bag_path,'r')
        self.odom = []
        for _,msg,_ in bag.read_messages(topics=['/X1/odom']):
            self.odom.append(msg)

        # Keep track of where in the odometry bag we are
        self.step_idx = 0

        # Action space: {left,up&left,up,up&right,right,down&right,down,down&left,no action}
        self.action_space = spaces.Discrete(9)
        r22 = np.sqrt(2)/2
        self.actions = [(-1,0),(-r22,r22),(0,1),(r22,r22),(1,0),(r22,-r22),(0,-1),(-r22,-r22),(0,0)]

        # How far (in radians) to move the camera each step
        self.action_step = action_step

        # State space is defined as the pan and tilt angles 
        self.camera_angles = np.zeros(2) # [0,0] means pointing along robot heading
        self.observation_space = spaces.Box(np.float32([-np.pi,-np.pi]),np.float32([np.pi,np.pi]))

    def reset(self):
        '''
        Reset the gym environment
        Set the robot back to the beginning of the odometry bag
        Return the observation from the start of the episode
        '''
        # Put robot back at beginning of odometry bag
        self.step_idx = 0

        # Create new octomap instance bc OcTree.clear() causes seg fault
        del self.tree_new
        self.tree_new = octomap.OcTree(self.octomap_res)

    def step(self,action):
        '''
        Take a step in the simulation episode
        Move the camera according to action
        Return (state,reward,done,info)
        '''
        pose = self.get_pose(self.step_idx)
        num_voxels = self.tree_new.getNumLeafNodes()
        self.camera_angles += self.action_step * self.actions[action]

    def raycast_view(self,pose):
        '''
        Simulate a camera view from robots pose and camera_angles
        Return a point cloud assuming no noise
        '''

        # Robot position
        origin = pose[:3]
        q = pose[3:]

        # Offset camera pan by robot heading (assuming robot on XY plane)
        theta = self.camera_angles[0] + 2*np.arccos(q[-1])
        phi = self.camera_angles[1]

        # Basis vectors for camera plane
        v = np.array([1.,0.,0.])
        e1 = np.array([0.,0.,1.])
        e2 = np.array([0.,-1.,0.])

        r1 = R.from_quat([e1[0]*np.sin(theta/2),e1[1]*np.sin(theta/2),e1[2]*np.sin(theta/2),np.cos(theta/2)])
        v = r1.apply(v)
        e2 = r1.apply(e2)
        r2 = R.from_quat([e2[0]*np.sin(phi/2),e2[1]*np.sin(phi/2),e2[2]*np.sin(phi/2),np.cos(phi/2)])
        v = r2.apply(v)
        e1 = r2.apply(e1)

        corners = np.zeros((4,3))
        n = 0
        for dtheta in np.linspace(-self.camera_fov[0],self.camera_fov[0],2):
            for dphi in np.linspace(-self.camera_fov[1],self.camera_fov[1],2):
                rr1 = R.from_quat([e1[0]*np.sin(dtheta/2),e1[1]*np.sin(dtheta/2),e1[2]*np.sin(dtheta/2),np.cos(dtheta/2)])
                vec = rr1.apply(v)
                e22 = rr1.apply(e2)
                rr2 = R.from_quat([e22[0]*np.sin(dphi/2),e22[1]*np.sin(dphi/2),e22[2]*np.sin(dphi/2),np.cos(dphi/2)])
                vec = rr2.apply(vec)

                corners[n,:] = vec
                n+=1
        
        theta_lin = np.linspace(corners[0],corners[1],self.camera_res[0])
        phi_lin = np.linspace(corners[0],corners[2],self.camera_res[1])
        cloud = []
        for t in theta_lin:
            for p in phi_lin:
                end = np.zeros(3)
                v = t + (p-phi_lin[0])
                hit = self.tree_gt.castRay(origin=origin,direction=v,end=end,maxRange=self.camera_range)
                if hit:
                    cloud.append(end)
        
        return np.unique(np.array(cloud),axis=0)
        

    def get_pose(self,idx):
        '''
        Return the pose as a numpy array for a given idx in the odometry bag
        (x,y,z,q_x,q_y,q_z,q_w)
        '''
        msg = self.odom[idx]
        pose = np.zeros(7,dtype=np.double)

        pose[0] = msg.pose.pose.position.x
        pose[1] = msg.pose.pose.position.y
        pose[2] = msg.pose.pose.position.z
        pose[3] = msg.pose.pose.orientation.x
        pose[4] = msg.pose.pose.orientation.y
        pose[5] = msg.pose.pose.orientation.z
        pose[6] = msg.pose.pose.orientation.w

        return pose

    def get_state(self):
        '''
        Return the current state of the simulation
        '''
        pass
