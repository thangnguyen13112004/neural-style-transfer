import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import copy
import io

from ThuVien import image_loader, st_imshow, get_style_model_and_losses, run_style_transfer, StyleLoss, ContentLoss, Normalization, get_input_optimizer


# Thiết lập trang
st.set_page_config(
    page_title="Thử Nghiệm Neural Style Transfer",
    page_icon="❄️",
    layout="wide"
)

# Đặt device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Kích thước ảnh
imsize = 512 if torch.cuda.is_available() else 128 

# Biến điều chỉnh hàm run_style_transfer
num_steps = 300

# Khởi tạo giá trị chuẩn hóa mô hình VGG19 từ ImageNet
cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

# Tải mô hình VGG19
@st.cache_resource
def load_model():
    model = models.vgg19_bn(pretrained=True).features.to(device).eval()
    return model

cnn = load_model()

# Phân tích cấu trúc mô hình VGG19 và trả về thông tin các layer
def analyze_model(model):
    conv_layers = []
    pool_layers = []
    norm_layers = []
    block_info = []
    
    i = 0   # Đếm số lượng Conv layer
    j = 0   # Block VGG19
    current_block = []
    
    prev_layer = None
    
    for layer in model.children():
        if isinstance(layer, nn.Conv2d):
            # Nếu là Conv đầu tiên của một block, bắt đầu block mới
            if i == 0 or isinstance(prev_layer, (nn.AvgPool2d, nn.MaxPool2d)):
                j += 1
                if current_block:
                    block_info.append((f"Block {j-1}", current_block.copy()))
                current_block = []
            
            i += 1
            name = f'conv{i}'
            conv_layers.append(name)
            current_block.append(name)
            
        elif isinstance(layer, nn.BatchNorm2d):
            name = f'bn_{i}'
            norm_layers.append(name)
            current_block.append(name)
            
        elif isinstance(layer, nn.ReLU):
            name = f'relu_{i}'
            current_block.append(name)
            
        elif isinstance(layer, (nn.MaxPool2d, nn.AvgPool2d)):
            name = f'pool_{i}'
            pool_layers.append(name)
            current_block.append(name)
        
        prev_layer = layer
    
    # Thêm block cuối cùng
    if current_block:
        block_info.append((f"Block {j}", current_block.copy()))
    
    return conv_layers, pool_layers, norm_layers, block_info

# Hàm chạy style transfer
def run_style_transfer(model, style_losses, content_losses, input_img, 
                      num_steps, style_weight=1000000, content_weight=1):
    optimizer = get_input_optimizer(input_img)
    
    # Thêm biến theo dõi best loss và best image
    best_loss = float('inf')
    best_img = input_img.clone()
    
    run = [0]
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    while run[0] <= num_steps:
        
        def closure():
            nonlocal best_loss, best_img  # Thêm nonlocal để cập nhật biến bên ngoài
            input_img.data.clamp_(0, 1)
            
            optimizer.zero_grad()
            model(input_img)
            
            style_score = 0
            content_score = 0
            
            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss
                
            style_score *= style_weight
            content_score *= content_weight
            
            loss = style_score + content_score
            loss.backward()
            
            run[0] += 1
            progress = min(run[0] / num_steps, 1.0)
            progress_bar.progress(progress)
            status_text.text(f"Bước {run[0]}/{num_steps}: Style Loss: {style_score.item():4f} Content Loss: {content_score.item():4f}")
            
            # Thêm phần lưu ảnh tốt nhất
            with torch.no_grad():
                total_loss = loss.item()
                if total_loss < best_loss:
                    best_loss = total_loss
                    best_img = input_img.detach().clone()
                
            return style_score + content_score
        
        optimizer.step(closure)
    
    input_img.data.clamp_(0, 1)
    progress_bar.empty()
    status_text.empty()
    
    # Hiển thị best loss cuối cùng
    st.write(f"Best loss: {best_loss}")
    
    # Trả về ảnh tốt nhất thay vì ảnh cuối cùng
    return best_img
# Xây dựng mô hình tùy chỉnh cho style transfer
def get_style_model_and_losses(cnn, cnn_normalization_mean, cnn_normalization_std,
                               style_img, content_img,
                               content_layers, style_layers, norm_type, pool_type):
    cnn = copy.deepcopy(cnn)
    
    # Ghi log quá trình xây dựng mô hình
    log_output = io.StringIO()
    
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
    
    prev_layer = None
    
    print("Cấu trúc mô hình được xây dựng:", file=log_output)
    
    for layer in cnn.children():
        # Kiểm tra loại layer
        if isinstance(layer, nn.Conv2d):
            # Nếu là Conv đầu tiên của một block, in ra Block mới
            if i == 0 or isinstance(prev_layer, (nn.AvgPool2d, nn.MaxPool2d)):
                j += 1
                print(f"\n**Block {j}:**", file=log_output)
            
            i += 1 
            name = f'conv{i}'
            print(f"{name} ->", end=" ", file=log_output)
            
        elif isinstance(layer, nn.BatchNorm2d):
            name = f'norm_{i}'
            print(f"{name} ->", end=" ", file=log_output)
            
            # Thay thế batch norm bằng loại chuẩn hóa được chọn
            if norm_type == "Instance Normalization":
                layer = nn.InstanceNorm2d(layer.num_features)
            elif norm_type == "Group Normalization":
                layer = nn.GroupNorm(num_groups=32, num_channels=layer.num_features)
            # Giữ nguyên BatchNorm nếu được chọn
            
        elif isinstance(layer, nn.ReLU):
            name = f'relu_{i}'
            print(f"{name}", file=log_output)
            layer = nn.ReLU(inplace=False)
            
        elif isinstance(layer, nn.MaxPool2d):
            name = f'pool_{i}'
            print(f"{name}", file=log_output)
            
            # Thay thế Max pooling bằng Average pooling nếu được chọn
            if pool_type == "Average Pooling":
                layer = nn.AvgPool2d(kernel_size=layer.kernel_size, stride=layer.stride, padding=layer.padding)
            # Giữ nguyên MaxPool nếu được chọn
            
        else:
            raise RuntimeError(f'Không hỗ trợ layer: {layer.__class__.__name__}')
    
        prev_layer = layer
        model.add_module(name, layer)
    
        if name in content_layers:
            # thêm content loss:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module(f"content_loss_{i}", content_loss)
            content_losses.append(content_loss)
            print(f"Thêm content loss tại {name}", file=log_output)
    
        if name in style_layers:
            # thêm style loss:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module(f"style_loss_{i}", style_loss)
            style_losses.append(style_loss)
            print(f"Thêm style loss tại {name}", file=log_output)
    
    # Cắt bỏ các lớp sau ContentLoss và StyleLoss cuối cùng
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break
    
    model = model[:(i + 1)]
    print("\nMô hình đã được cắt tại", i+1, file=log_output)
    
    return model, style_losses, content_losses, log_output.getvalue()

# Phân tích cấu trúc mô hình
conv_layers, pool_layers, norm_layers, block_info = analyze_model(cnn)

st.title("Thử Nghiệm Neural Style Transfer")
st.write("Tùy chỉnh các tham số của thuật toán Neural Style Transfer.")

# Tạo hai cột cho phần upload ảnh
col_upload1, col_upload2 = st.columns(2)

with col_upload1:
    # Upload ảnh nội dung
    content_file = st.file_uploader("Tải lên ảnh nội dung", type=["jpg", "jpeg", "png"])

with col_upload2:
    # Upload ảnh phong cách
    style_file = st.file_uploader("Tải lên ảnh phong cách", type=["jpg", "jpeg", "png"])

# Xử lý khi người dùng đã tải lên cả 2 ảnh
if content_file and style_file:
    content_img = image_loader(content_file, imsize)
    style_img = image_loader(style_file, imsize)
    
    # Kiểm tra kích thước
    if content_img.size() != style_img.size():
        st.error("Cần cùng kích thước giữa 2 ảnh!")
    else:
        # Hiển thị ảnh gốc
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Ảnh nội dung")
            content_image = st_imshow(content_img, "")
            st.image(content_image, use_column_width=True)
        
        with col2:
            st.subheader("Ảnh phong cách")
            style_image = st_imshow(style_img, "")
            st.image(style_image, use_column_width=True)
        
        st.subheader("Cấu trúc VGG19")
        
        # Hiển thị thông tin về các block và layers
        tabs = st.tabs([block[0] for block in block_info])
        for i, tab in enumerate(tabs):
            with tab:
                st.write(f"Layers trong {block_info[i][0]}:")
                st.write(", ".join(block_info[i][1]))
        
        st.subheader("Tùy chỉnh cài đặt")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("Chọn Content Layers")
            content_layers = st.multiselect(
                "Content Layers",
                options=conv_layers,
                default=['conv4']
            )
        
        with col2:
            st.write("Chọn Style Layers")
            style_layers = st.multiselect(
                "Style Layers",
                options=conv_layers,
                default=['conv1', 'conv3', 'conv5', 'conv9', 'conv13']
            )
        
        with col3:
            st.write("Tùy chỉnh tham số")
            num_steps = st.slider("Số bước tối ưu", 50, 500, 300)
            style_weight = st.slider("Style Weight", 1000, 1000000, 1000000, step=1000)
            content_weight = st.slider("Content Weight", 1, 100, 1)
        
        col1, col2 = st.columns(2)
        
        with col1:
            norm_type = st.selectbox(
                "Loại chuẩn hóa",
                ["Batch Normalization", "Instance Normalization", "Group Normalization"]
            )
        
        with col2:
            pool_type = st.selectbox(
                "Loại Pooling",
                ["Max Pooling", "Average Pooling"]
            )
        
        # Nút để bắt đầu chuyển đổi phong cách
        if st.button("Bắt đầu chuyển đổi phong cách", use_container_width=True):
            if not content_layers:
                st.error("Vui lòng chọn ít nhất một Content Layer!")
            elif not style_layers:
                st.error("Vui lòng chọn ít nhất một Style Layer!")
            else:
                # Khởi tạo ảnh đầu vào
                input_img = content_img.clone()  # Sử dụng ảnh nội dung làm khởi tạo
                
                with st.expander("Xem quá trình xây dựng mô hình", expanded=False):
                    # Lấy các giá trị cần thiết
                    model, style_losses, content_losses, model_log = get_style_model_and_losses(
                        cnn, cnn_normalization_mean, cnn_normalization_std,
                        style_img, content_img, content_layers, style_layers,
                        norm_type, pool_type
                    )
                    st.code(model_log)
                
                st.subheader("Quá trình tối ưu")
                
                # Khởi chạy mô hình
                output = run_style_transfer(
                    model, style_losses, content_losses,
                    input_img, num_steps=num_steps,
                    style_weight=style_weight, content_weight=content_weight
                )
                
                st.subheader("Kết quả")
                output_image = st_imshow(output, "")
                st.image(output_image, caption="Ảnh kết quả")
                
                # Nút tải xuống kết quả
                buf = io.BytesIO()
                output_pil = transforms.ToPILImage()(output.cpu().squeeze(0))
                output_pil.save(buf, format="PNG")
                byte_im = buf.getvalue()
                
                st.download_button(
                    label="Tải xuống ảnh kết quả",
                    data=byte_im,
                    file_name="neural_style_output.png",
                    mime="image/png",
                    use_container_width=True
                )
else:
    st.info("Vui lòng tải lên cả ảnh nội dung và ảnh phong cách để tiếp tục")