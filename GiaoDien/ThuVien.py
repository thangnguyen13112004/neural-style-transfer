import torch
#Chứa các lớp giúp xây dựng mạng nơ-ron như nn.Module, nn.Conv2d, nn.ReLU, nn.Sequential, v.v.
import torch.nn as nn

#Chứa các hàm tiện ích như:
#    F.mse_loss(): Hàm mất mát bình phương trung bình (Mean Squared Error).
#    F.relu(): Hàm kích hoạt ReLU.
#    F.softmax(): Hàm chuẩn hóa softmax.
import torch.nn.functional as F

#Chứa các thuật toán tối ưu như:
#    optim.LBFGS(): Thuật toán tối ưu hóa đặc biệt được dùng trong Style Transfer.
import torch.optim as optim

#Thư viện xử lý ảnh
#    PIL (Pillow) là thư viện để mở, xử lý và lưu ảnh.
#    Được dùng để đọc ảnh đầu vào trong image_loader().
from PIL import Image

#Thư viện vẽ đồ thị
import matplotlib.pyplot as plt

#Dùng để tiền xử lý ảnh, chuyển đổi ảnh thành tensor và chuẩn hóa dữ liệu.
import torchvision.transforms as transforms

#Dùng để tải mô hình VGG19 đã được huấn luyện trước trên ImageNet.
import torchvision.models as models

import streamlit as st

#Dùng để sao chép mô hình VGG19 trước khi chỉnh sửa các lớp, tránh ảnh hưởng đến mô hình gốc.
import copy
#------------------------------------------------------------------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#-----------------------------------------------------CÁC HÀM CẦN THIẾT------------------------------------------
# Thay đổi kích thước & chuyển đổi ảnh thành tensor PyTorch.
def image_loader(image, imsize):
    loader = transforms.Compose([
        transforms.Resize((imsize, imsize)),  
        transforms.ToTensor()
    ])  
    # Xử lý ảnh (có thể là file được tải lên hoặc đường dẫn)
    if isinstance(image, str):
        image = Image.open(image)
    else:
        # Nếu là file được upload từ Streamlit
        image = Image.open(image)
    # unsqueeze: tensor có kích thước (1, C, H, W).
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

def st_imshow(tensor, title=None):
    unloader = transforms.ToPILImage()
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    if title:
        st.subheader(title)
    return image 

# gram_matrix cho style loss
def gram_matrix(input):
    a, b, c, d = input.size()  
    # a=batch size(=1)
    # b=số lượng feature maps
    # (c,d)=Chiều cao và chiều rộng của mỗi feature map. Tổng số điểm ảnh trong một kênh= c x d

    features = input.view(a * b, c * d)  # Chuyển tensor từ dạng (1, C, H, W) thành (C, H × W).

    G = torch.mm(features, features.t())  # Tính Gram Matrix

    # chuẩn hóa giúp giữ giá trị ổn định, tránh sai số quá lớn.
    return G.div(a * b * c * d)

# StyleLoss
class StyleLoss(nn.Module):

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input

# content loss 
class ContentLoss(nn.Module):

    def __init__(self, target):
        super(ContentLoss, self).__init__()
        # Ngắt kết nối target khỏi cây tính toán gradient, vì target chỉ là tham chiếu cố định, 
        # không cần cập nhật trong quá trình tối ưu.
        # Nếu không có .detach(), PyTorch sẽ cố gắng tính gradient cho target, gây lỗi.
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input

# chuẩn hóa ảnh phù hợp mô hình VGG19
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        # normalize ảnh
        return (img - self.mean) / self.std
    
# ảnh đầu vào (input_img) được cập nhật trong quá trình tối ưu hóa phong cách.
def get_input_optimizer(input_img):
    # tính gradient của input_img bằng LBFGS giúp cập nhật pixel ảnh trong quá trình tối ưu hóa.
    optimizer = optim.LBFGS([input_img.requires_grad_()])
    return optimizer

# khởi tạo lấy lớp tích chập mong muốn để tính loss 
#content_layers_default = ['conv4_2']
#style_layers_default = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']


content_layers_default = ['conv4']
style_layers_default = ['conv1', 'conv3', 'conv5', 'conv9', 'conv13']

# Xây dựng mô hình mới từ VGG19, trả về model, style_losses, content_losses
def get_style_model_and_losses(cnn, cnn_normalization_mean, cnn_normalization_std,
                               style_img, content_img,
                               content_layers=content_layers_default,
                               style_layers=style_layers_default):
    cnn = copy.deepcopy(cnn)

    # normalization
    normalization = Normalization(cnn_normalization_mean, cnn_normalization_std).to(device)

    # Khởi tạo danh sách lưu các lớp mất mát
    content_losses = []
    style_losses = []

    # Xây dựng mô hình mới từ VGG19
    # Thêm lớp Normalization vào đầu mô hình.
    model = nn.Sequential(normalization)

    i = 0   # Đếm số lượng Conv layer
    j = 0   # Block VGG19

    for layer in cnn.children():
        #kiểm tra loại
        if isinstance(layer, nn.Conv2d):
            # Nếu là Conv đầu tiên của một block, in ra Block mới
            if i == 0 or isinstance(prev_layer, (nn.AvgPool2d, nn.MaxPool2d)):
                j += 1
                print(f"\n **Block {j}:**")
            
            i += 1 
            name = 'conv{}'.format(i)
            print(f"{name} ->", end=" ")
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'in_{}'.format(i)
            print(f"{name} ->", end=" ")
            # thay bằng InstanceNorm2d
            layer = nn.InstanceNorm2d(layer.num_features)
            #Thay thế bằng GroupNorm (chia thành 32 nhóm, có thể điều chỉnh)
            #layer = nn.GroupNorm(num_groups=32, num_channels=layer.num_features)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            print(f"{name} \n")
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
            print(f"{name} \n")
            # thay thế Max pooling bằng Average pooling
            #layer = nn.AvgPool2d(kernel_size=layer.kernel_size, stride=layer.stride, padding=layer.padding)
        else:
            raise RuntimeError('không có layer: {}'.format(layer.__class__.__name__))

        prev_layer = layer
        model.add_module(name, layer)

        if name in content_layers:
            # thêm content loss:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)


        if name in style_layers:
            # thêm style loss:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    # Cắt bỏ các lớp sau ContentLoss và StyleLoss cuối cùng, giữ lại các layer cần thiết.
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]

    return model, style_losses, content_losses

# hàm chính: neural style transfer
def run_style_transfer(model, style_losses, content_losses, input_img, num_steps,
                       style_weight=1000000, content_weight=1):
    """Run the style transfer."""
    optimizer = get_input_optimizer(input_img)

    best_loss = float('inf')
    best_img = input_img.clone()

    print('Bản tính toán loss..')
    run = [0]
    while run[0] <= num_steps:
        #L-BFGS sẽ gọi closure() nhiều lần trong một bước để tìm hướng tối ưu tốt nhất.
        def closure():
            nonlocal best_loss, best_img
            # cập nhật cho input image
            input_img.data.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)

            style_score = 0
            content_score = 0

            #nhiều layers
            for style_layer in style_losses:
                style_score += (1/5)*style_layer.loss

            #1 layer
            for content_layer in content_losses:
                content_score += content_layer.loss

            style_score *= style_weight #(alpha)
            content_score *= content_weight #(beta) 

            loss = style_score + content_score
            #Tính gradient để cập nhật input_img
            loss.backward()
    
            run[0] += 1
            #50 epoch mỗi lần
            if run[0] % 50 == 0:
                print("run {}:".format(run))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                    style_score.item(), content_score.item()))
                print()
            # Cập nhật ảnh tốt nhất
            with torch.no_grad():
                total_loss = loss.item()
                if total_loss < best_loss:
                    best_loss = total_loss
                    best_img = input_img.detach().clone()

            return loss
            
        optimizer.step(closure)
        
    print("Best loss:", best_loss) 
    # Đảm bảo giá trị pixel nằm trong [0, 1]
    best_img.data.clamp_(0, 1)

    return best_img
