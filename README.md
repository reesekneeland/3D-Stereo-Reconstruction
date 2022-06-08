# 3D-Stereo-Reconstruction
A computer vision stereo reconstruction algorithm that creates a 3D map given two images.

### Sift Feature Matching
![1](https://user-images.githubusercontent.com/77468346/172683005-9f4a1078-bd0a-4a93-a77e-9d9dbd3b7428.png)
**find_match()** visualization for SIFT feature mapping between img1 and img2, using 0.75 threshold for neighbor search.

### Fundemental Matrix Computation
![2](https://user-images.githubusercontent.com/77468346/172683025-aa7d098e-d652-48b0-a6ae-98ce38fe19dc.png)


**compute_F()** computes the fundamental matrix by using an 8-point 1000 iteration RANSAC algorithm, a null_space calculator, and an SVD function. Epipolar lines are pictured above.

### Triangulation
![3](https://user-images.githubusercontent.com/77468346/172683034-8dbfc8ea-686c-4226-9ec4-a2ad0d9dd5f3.png)
**triangulation()** and **skew_pt()** combine to provide the triangulated 3d point representation of the camera poses, of which **disambiguate_pose()**  returns the best one.

### Stereo Rectification
![4](https://user-images.githubusercontent.com/77468346/172683041-aa838980-a775-40b3-81c6-8be37968ad4d.png)
 **compute_rectification()** rectifies the disambiguated camera poses and computes the dense stereo matching between the two views based on the calculated homographies.
 
 ![5](https://user-images.githubusercontent.com/77468346/172683051-8189250c-2b69-4803-9e20-f185c5f92bc0.png)
Finally, we calculate the disparity map using **dense_match()**, making use of dense SIFT feature matching across epipolar lines, I used a combined representation of SIFT features with a keypoint size of 3 and 10, with the size 10 SIFT features weighted higher. The disparity is then calculated as a difference in the rectified positions, represented in the disparity map above.
