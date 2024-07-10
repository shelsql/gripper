from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import cv2

from matplotlib.patches import ConnectionPatch

from torchmetrics.functional import pairwise_cosine_similarity

from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA

import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

def vis_rgbs(rgb, name):
    rgb = rgb.permute(1, 2, 0).cpu().numpy()
    print("Visualizing image", name, rgb.shape, np.max(rgb), np.min(rgb), np.mean(rgb))
    rgb = (rgb - np.min(rgb)) / (np.max(rgb) - np.min(rgb))
    rgb *= 255
    cv2.imwrite("./match_demo_vis/cropped_rgb_" + name + ".png", rgb)
      
class Dinov2Matcher:

  def __init__(self, repo_name="facebookresearch/dinov2", model_name="dinov2_vitl14_reg", smaller_edge_size=448, half_precision=False, device="cuda:3"):
    self.repo_name = repo_name
    self.model_name = model_name
    self.smaller_edge_size = smaller_edge_size
    self.half_precision = half_precision
    self.device = device

    if self.half_precision:
      #self.model = torch.hub.load(repo_or_dir=repo_name, model=model_name).half().to(self.device)
      self.model = torch.hub.load(repo_or_dir="../dinov2",source="local", model=model_name, pretrained=False).half().to(device)
      self.model.load_state_dict(torch.load('./dinov2_weights/dinov2_vitl14_reg4_pretrain.pth'))
    else:
      #self.model = torch.hub.load(repo_or_dir=repo_name, model=model_name).to(self.device)
      self.model = torch.hub.load(repo_or_dir="../dinov2",source="local", model=model_name, pretrained=False).to(device)
      self.model.load_state_dict(torch.load('./dinov2_weights/dinov2_vitl14_reg4_pretrain.pth'))

    self.model.eval()

    self.transform = transforms.Compose([
        transforms.Resize(size=smaller_edge_size, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), # imagenet defaults
      ])

  # https://github.com/facebookresearch/dinov2/blob/255861375864acdd830f99fdae3d9db65623dafe/notebooks/features.ipynb
  def prepare_image(self, rgb_image_numpy):
    #print(mask_numpy.shape, mask_numpy.dtype)
    image = Image.fromarray(rgb_image_numpy)
    image_tensor = self.transform(image)
    resize_scale = image.width / image_tensor.shape[2]

    # Crop image to dimensions that are a multiple of the patch size
    height, width = image_tensor.shape[1:] # C x H x W
    cropped_width, cropped_height = width - width % self.model.patch_size, height - height % self.model.patch_size # crop a bit from right and bottom parts
    image_tensor = image_tensor[:, :cropped_height, :cropped_width]

    grid_size = (cropped_height // self.model.patch_size, cropped_width // self.model.patch_size)
    return image_tensor, grid_size, resize_scale
  
  def prepare_mask(self, mask_image_numpy, grid_size, resize_scale):
    cropped_mask_image_numpy = mask_image_numpy[:int(grid_size[0]*self.model.patch_size*resize_scale), :int(grid_size[1]*self.model.patch_size*resize_scale)]
    image = Image.fromarray(cropped_mask_image_numpy)
    resized_mask = image.resize((grid_size[1], grid_size[0]), resample=Image.Resampling.NEAREST)
    resized_mask = np.asarray(resized_mask).flatten()
    return resized_mask
  
  def extract_features(self, image_tensor):
    with torch.inference_mode():
      if self.half_precision:
        image_batch = image_tensor.unsqueeze(0).half().to(self.device)
      else:
        image_batch = image_tensor.unsqueeze(0).to(self.device)

      tokens = self.model.get_intermediate_layers(image_batch)[0].squeeze()
    return tokens.cpu().numpy()
  
  def idx_to_source_position(self, idx, grid_size, resize_scale):
    row = (idx // grid_size[1])*self.model.patch_size*resize_scale + self.model.patch_size / 2
    col = (idx % grid_size[1])*self.model.patch_size*resize_scale + self.model.patch_size / 2
    return row, col
  
  def get_embedding_visualization(self, tokens, grid_size, resized_mask=None):
    pca = PCA(n_components=3)
    if resized_mask is not None:
      tokens = tokens[resized_mask]
    reduced_tokens = pca.fit_transform(tokens.astype(np.float32))
    print(np.max(reduced_tokens), np.min(reduced_tokens), reduced_tokens.shape)
    if resized_mask is not None:
      tmp_tokens = np.zeros((*resized_mask.shape, 3), dtype=reduced_tokens.dtype)
      tmp_tokens[resized_mask] = reduced_tokens
      reduced_tokens = tmp_tokens
    reduced_tokens = reduced_tokens.reshape((*grid_size, -1))
    normalized_tokens = (reduced_tokens-np.min(reduced_tokens))/(np.max(reduced_tokens)-np.min(reduced_tokens))
    return normalized_tokens
  
  
  def vis_features(self, images, feat_masks, feats, i):
    C, H, W = images.shape
    N_tokens, C_feat = feats.shape
    H_feat = H //14
    W_feat = W //14
    images = images.permute(1, 2, 0).cpu().numpy() # H, W, 3
    #feats = feats.reshape(H_feat, W_feat, C_feat)# H*W, 1024
    reshaped_masks = feat_masks
    feat_masks = feat_masks.reshape(H_feat, W_feat)
    print(images.shape, feats.shape, feat_masks.shape)
    pca = PCA(n_components=3)
    feat_map = np.zeros((H_feat, W_feat, 3))
    feats_i = feats[reshaped_masks > 0].astype(np.float32)
    print(np.max(feats_i), np.min(feats_i), np.mean(feats_i))
    pca_feats = pca.fit_transform(feats_i) # H*W, 3
    print(np.max(pca_feats), np.min(pca_feats), pca_feats.shape)
    pca_feats = (pca_feats - np.min(pca_feats)) / (np.max(pca_feats) - np.min(pca_feats))
    #print(np.max(pca_feats), np.min(pca_feats), pca_feats.shape)
    feat_map[feat_masks > 0] = pca_feats
    feat_map = feat_map*255
    cv2.imwrite("./match_demo_vis/ref_feat_%.2d.png" % i, feat_map)



# Load image and mask
image1 = cv2.cvtColor(cv2.imread('./rendered_without_env/000_rgb.png', cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
mask1 = cv2.imread('./rendered_without_env/000_id.exr', cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)[:,:,0] > 0
a = np.where(mask1 > 0)
image1 = image1[np.min(a[0]):np.max(a[0])+1, np.min(a[1]):np.max(a[1])+1]
mask1 = mask1[np.min(a[0]):np.max(a[0])+1, np.min(a[1]):np.max(a[1])+1]

# Init Dinov2Matcher
dm = Dinov2Matcher(half_precision=False)

# Extract features
image_tensor1, grid_size1, resize_scale1 = dm.prepare_image(image1)
features1 = dm.extract_features(image_tensor1)
print(features1.shape, features1.max(), features1.min(), features1.mean())
for i in range(10):
  print(features1[i,:10])
  pass

# Visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,20))
ax1.imshow(image1)
resized_mask = dm.prepare_mask(mask1, grid_size1, resize_scale1)
#print("resized mask:", resized_mask.shape, resized_mask.sum())
vis_image = dm.get_embedding_visualization(features1, grid_size1, resized_mask)
print("Normalized tokens:",vis_image.shape)
'''
for i in range(0, vis_image.shape[0], 2):
  for j in range(0, vis_image.shape[1], 2):
    print(vis_image[i,j])
  print()
'''
ax2.imshow(vis_image)
fig.tight_layout()
plt.savefig("./match_demo_vis/img_and_feats.png")

# More info
print("image1.shape:", image1.shape)
print("mask1.shape:", mask1.shape)
print("image_tensor1.shape:", image_tensor1.shape)
print("grid_size1:", grid_size1)
print("resize_scale1:", resize_scale1)


# Extract image2 features
image2 = cv2.cvtColor(cv2.imread('./../../datasets/Ty_data/1_D415_front_0/000000.png', cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
mask2 = cv2.imread('./../../datasets/Ty_data/1_D415_front_0/000000_mask.exr', cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)[:,:,0] >= 9
a = np.where(mask2 > 0)
image2 = image2[np.min(a[0]):np.max(a[0])+1, np.min(a[1]):np.max(a[1])+1]
mask2 = mask2[np.min(a[0]):np.max(a[0])+1, np.min(a[1]):np.max(a[1])+1]
image_tensor2, grid_size2, resize_scale2 = dm.prepare_image(image2)
print(image_tensor2.max(), image_tensor2.min(), image_tensor2.mean())
features2 = dm.extract_features(image_tensor2)
print(features2.shape, features2.max(), features2.min(), features2.mean())

# Build knn using features from image1, and query all features from image2
'''
knn = NearestNeighbors(n_neighbors=1)
knn.fit(features1)
distances, match2to1 = knn.kneighbors(features2)
match2to1 = np.array(match2to1)
'''
for n in range(32):
  # Extract image1 features
  image1 = cv2.cvtColor(cv2.imread('./rendered_without_env/%.3d_rgb.png' % n, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
  mask1 = cv2.imread('./rendered_without_env/%.3d_id.exr' % n, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)[:,:,0] > 0
  
  #image1 = cv2.cvtColor(cv2.imread('./../../datasets/Ty_data/1_D415_front_0/000200.png', cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
  #mask1 = cv2.imread('./../../datasets/Ty_data/1_D415_front_0/000200_mask.exr', cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)[:,:,0] >= 9
  #image1 = cv2.cvtColor(cv2.imread('./../../datasets/Ty_data/1_D415_front_0/000100.png', cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
  #mask1 = cv2.imread('./../../datasets/Ty_data/1_D415_front_0/000100_mask.exr', cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)[:,:,0] >= 9
  a = np.where(mask1 > 0)
  image1 = image1[np.min(a[0]):np.max(a[0])+1, np.min(a[1]):np.max(a[1])+1]
  mask1 = mask1[np.min(a[0]):np.max(a[0])+1, np.min(a[1]):np.max(a[1])+1]
  #image1 = image1[::-1]
  #mask1 = mask1[::-1].astype(np.uint8)
  #image1 = cv2.resize(image1, , interpolation=cv2.INTER_LINEAR)
  #mask1 = cv2.resize(mask1, dsize=(0,0), fx=0.15, fy=0.15, interpolation=cv2.INTER_NEAREST) >0
  image_tensor1, grid_size1, resize_scale1 = dm.prepare_image(image1)
  vis_rgbs(image_tensor1, str(n))
  features1 = dm.extract_features(image_tensor1)
  resized_mask = dm.prepare_mask(mask1, grid_size1, resize_scale1)
  dm.vis_features(image_tensor1, resized_mask, features1, n)
  print(features1.shape, features1.max(), features1.min(), features1.mean())

  cosine_sims = pairwise_cosine_similarity(torch.tensor(features1), torch.tensor(features2))
  #print(cosine_sims.shape)
  #print(cosine_sims.max(), cosine_sims.min())
  matches = torch.nonzero(cosine_sims > 0.8)
  #print(matches.shape, matches[:10])

  #plt.plot(sorted(distances))
  #plt.savefig("./match_demo_vis/dists.png")

  fig = plt.figure(figsize=(20,10))
  ax1 = fig.add_subplot(121)
  ax2 = fig.add_subplot(122)

  ax1.imshow(image1)

  ax2.imshow(image2)

  print("Mask shapes:", mask1.shape, mask2.shape)

  for i in range(matches.shape[0]):
    idx1 = matches[i,0]
    idx2 = matches[i,1]
    row, col = dm.idx_to_source_position(idx1, grid_size1, resize_scale1)
    xyA = (col, row)
    if row >= mask1.shape[0] or col >= mask1.shape[1]: continue
    if not mask1[int(row), int(col)]: continue # skip if feature is not on the object

    row, col = dm.idx_to_source_position(idx2, grid_size2, resize_scale2)
    xyB = (col, row)
    if row >= mask2.shape[0] or col >= mask2.shape[1]: continue
    if not mask2[int(row), int(col)]: continue # skip if feature is not on the object

    #if np.random.rand() > 0.05: continue # sparsely draw so that we can see the lines...

    con = ConnectionPatch(xyA=xyB, xyB=xyA, coordsA="data", coordsB="data",
                          axesA=ax2, axesB=ax1, color=np.random.rand(3,))
    ax2.add_artist(con)
    
  plt.savefig("./match_demo_vis/matches_%d.png" % n)