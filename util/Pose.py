import transforms3d as t3d
import numpy as np


class Pose(object):
    """
        A class used to represent a Pose of an image (translation and rotation)

        Attributes
        ----------
        t: 1D np.array
            a 3x1 numpy float vector
        rotm_R : 2D np.array
            a 3x3 matrix for rotation matrix representation of the rotation
        quat_R : 1D np.array
            a 1x4 vector for the quaterion representation of the rotation
        midpoint : 1D np.array
            a centered translation vector
        affine: 2D np.array
            4x4 affine matrix representation
        """
    def __init__(self, t, rotm_R=None, quat_R=None):
        """
        :param t: (1D np.array) 1x3 translation vector
        :param rotm_R: (2D np.array) 3x3 rotation matrix, must be specified if quat_R is None
        :param quat_R: (1D np.array) 1x4 quaterion vector, must be specified if rotm_R is None
        :return: a Pose object
        """
        super(Pose, self).__init__()
        if t is None:
            self._set_nan_pose()
        else:
            assert (rotm_R is not None or quat_R is not None)
            self.quat_R = quat_R
            self.t = t
            self.rotm_R = rotm_R
            self.quat2rotm()
            self.rotm2quat()
            self.midpoint = None
            self.affine = None

    def _set_nan_pose(self):
        self.quat_R = np.ones(4) * np.nan
        self.t = np.ones(3) * np.nan
        self.rotm_R = np.ones((3, 3)) * np.nan

    def quat2rotm(self):
        """
        Initialize self.rotm_R by converting self.quat_R to a rotation matrix
        """
        if self.rotm_R is None and self.quat_R is not None:
            self.rotm_R = t3d.quaternions.quat2mat(self.quat_R / np.linalg.norm(self.quat_R))

    def rotm2quat(self):
        '''
        Initialize self.quat_R by converting self.rotm_R to a quaterion
        '''
        if self.rotm_R is not None and self.quat_R is None:
            self.quat_R = t3d.quaternions.mat2quat(self.rotm_R)

    def get_rotm_R(self):
        """
        Get the rotation matrix representation of the pose rotation (initialize if necessary)
        :return: the rotation matrix representation (self.rotm_R)
        """
        self.quat2rotm() # initialize from quaterion if necessary
        return self.rotm_R

    def get_affine(self):
        """
        Get the affine matrix representation of the pose (initialize if necessary)
        :return: self.affine
        """
        if self.affine is None:
            self.affine = t3d.affines.compose(self.t, self.rotm_R, np.ones(3))
        return self.affine

    def tovector(self):
        """
        Convert the pose into a vector representation
        :return: v, a 1x7 np.array where v[0:2] = the translation vector element and v[3:6] = the quaternion elements
        """
        v = np.zeros(7)
        v[0:3] = self.t[0:3]
        v[3:7] = self.quat_R[0:4]
        return v

    def rel_pose(self, other):
        """
        Calculate the relative pose between the pose (P) and another pose O such that PR = O
        :param other: (Pose) the relative pose is computed relative to this pose
        :return: the relative pose (Pose object)
        """
        rotm_R = np.dot(np.linalg.inv(self.rotm_R),other.rotm_R)
        t = other.t - self.t
        return Pose(t, rotm_R=rotm_R)

    def reverse(self):
        """
        :return: the reverse pose (-translation, inverse(orienration))
        """
        rotm_R = np.linalg.inv(self.rotm_R)
        t = -self.t
        return Pose(t, rotm_R=rotm_R)
