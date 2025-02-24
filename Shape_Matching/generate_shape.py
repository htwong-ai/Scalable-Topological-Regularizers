import torch

# Shape 1: circle
# Shape 2: two intersected circles
# Shape 3: boxplus
# Shape 4: sphere
def generate_shapes(shape_num, num_points, device = 'cpu'):
    half = int(num_points/2)
    
    if shape_num == 1:
        # make circle
        ref_shape = torch.randn(num_points,2, generator = torch.Generator().manual_seed(0))
        ref_shape = torch.nn.functional.normalize(ref_shape, p=2, dim=1) + torch.randn(num_points,2, generator = torch.Generator().manual_seed(2))*0.01 
        ref_shape = ref_shape.to(device)

    elif shape_num == 2:
        # make intersection of two circles
        ref_shape1 = torch.randn(half,2, generator = torch.Generator().manual_seed(0))
        ref_shape1 = torch.nn.functional.normalize(ref_shape1, p=2, dim=1) + torch.randn(half,2, generator = torch.Generator().manual_seed(2))*0.01 + torch.tensor([0.5,0.0])
        ref_shape2 = torch.randn(half,2, generator = torch.Generator().manual_seed(1))
        ref_shape2 = torch.nn.functional.normalize(ref_shape2, p=2, dim=1) + torch.randn(half,2, generator = torch.Generator().manual_seed(2))*0.01 - torch.tensor([0.5,0.0])
        ref_shape = torch.cat((ref_shape1, ref_shape2), 0).to(device)

    elif shape_num == 3:
        # make boxplus
        ref_shape1 = torch.rand(half, 2)*2 - 1
        ref_shape1[:, 1] = torch.randint(-1,2,(half,))
        ref_shape2 = torch.rand(half, 2)*2 - 1
        ref_shape2[:, 0] = torch.randint(-1,2,(half,))
        ref_shape = torch.cat((ref_shape1, ref_shape2), 0) + torch.randn(num_points,2)*0.01
        ref_shape = ref_shape.to(device)

    elif shape_num == 4:
        # make sphere
        ref_shape = torch.randn(num_points,3)
        ref_shape = torch.nn.functional.normalize(ref_shape, p=2, dim=1) + torch.randn(num_points,3)*0.01 
        ref_shape = ref_shape.to(device)

    return ref_shape
