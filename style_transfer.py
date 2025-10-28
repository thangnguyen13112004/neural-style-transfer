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

#Dùng để sao chép mô hình VGG19 trước khi chỉnh sửa các lớp, tránh ảnh hưởng đến mô hình gốc.
import copy
#------------------------------------------------------------------------------------------------------------------------

# Kiểm tra thiết bị
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   


#-----------------------------------------------------CÁC HÀM CẦN THIẾT------------------------------------------
# Thay đổi kích thước & chuyển đổi ảnh thành tensor PyTorch.
def image_loader(image_name, imsize):
    loader = transforms.Compose([
        transforms.Resize((imsize, imsize)),  
        transforms.ToTensor()
    ])  
    image = Image.open(image_name)
    # unsqueeze: tensor có kích thước (1, C, H, W).
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

# Hiển thị ảnh 
def imshow(tensor, title=None):
    unloader = transforms.ToPILImage()  # chuyển lại ảnh qua PIL
    image = tensor.cpu().clone()  # Sao chép tensor để tránh làm thay đổi dữ liệu gốc
    image = image.squeeze(0)      #  (C, H, W) để phù hợp với ToPILImage()
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)

def optimized_gram_matrix(input):
    batch_size, features, height, width = input.size()
    
    # Tối ưu việc reshape
    features_flat = input.reshape(batch_size * features, height * width)
    
    # Sử dụng torch.matmul thay vì torch.mm trong một số trường hợp có thể nhanh hơn
    gram = torch.matmul(features_flat, features_flat.t())
    
    # Thực hiện chuẩn hóa bằng phép nhân thay vì chia
    norm_factor = 1.0 / (batch_size * features * height * width)
    return gram * norm_factor
    
class EnhancedStyleLoss(nn.Module):
    def __init__(self, target_feature, content_weight=1.0, style_weight=1.0):
        super(EnhancedStyleLoss, self).__init__()
        self.target = optimized_gram_matrix(target_feature).detach()
        self.content_weight = content_weight
        self.style_weight = style_weight
        
    def forward(self, input):
        G = optimized_gram_matrix(input)
        # Style loss như bình thường
        style_loss = F.mse_loss(G, self.target) * self.style_weight
        
        # Thêm content loss để giữ cấu trúc (bố cục) tốt hơn
        content_loss = F.mse_loss(input, self.target_feature) * self.content_weight if hasattr(self, 'target_feature') else 0
        
        self.loss = style_loss + content_loss
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
            layer = nn.AvgPool2d(kernel_size=layer.kernel_size, stride=layer.stride, padding=layer.padding)
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
            style_loss = EnhancedStyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    # Cắt bỏ các lớp sau ContentLoss và StyleLoss cuối cùng, giữ lại các layer cần thiết.
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], EnhancedStyleLoss):
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



# ----------------------------------------------------BIẾN KHỞI TẠO--------------------------------------
# kích thước 512 nếu có GPU, else 128
imsize = 512 if torch.cuda.is_available() else 128 

# biến điều chỉnh hàm run_style_transfer
num_steps = 300

# khởi tạo Giá trị chuẩn hóa mô hình VGG19 từ ImageNet.
cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

image_directory = "../STYLE_TRANSFER/images/"
style_img = image_loader(image_directory + "color.JPG", imsize)
content_img = image_loader(image_directory + "blackwhite.jpg", imsize)

#-----------------------------------------------------------------------------------------------------------

# kiểm tra lỗi
assert style_img.size() == content_img.size(), "Chú ý: Cần cùng kích thước giữa 2 ảnh!"

#-----------------------------------------------------HÀM MAIN--------------------------------------
# Bật chế độ tương tác
plt.ion()

plt.figure()
imshow(style_img, title='Style Image')

plt.figure()
imshow(content_img, title='Content Image')


# tải mô hình VGG 19 để trích xuất đặc trưng từ ảnh (Convolutional + ReLU + Pooling).
cnn = models.vgg19_bn(pretrained=True).features.to(device).eval()

# khởi tạo ảnh nhiễu (input img)
input_img = torch.randn(content_img.data.size(), device=device)

plt.figure()
imshow(input_img, title='Input Image')


print('Mô hình style transfer dựa trên VGG19:')

# lấy các giá trị cần thiết
model, style_losses, content_losses = get_style_model_and_losses(cnn,
        cnn_normalization_mean, cnn_normalization_std, style_img, content_img)

# khởi chạy mô hình 
output = run_style_transfer(model, style_losses, content_losses,
                            input_img, num_steps=num_steps)

# Kết quả
plt.figure()
imshow(output, title='Output Image')

plt.ioff()
plt.show()