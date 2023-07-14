from flask import Flask, render_template, request
from PIL import Image
import torch
import torchvision.transforms as transforms

app = Flask(__name__)

# 신경망 모델 정의
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(32 * 8 * 8, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# 클래스 레이블
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

model_path = "/Users/sangeun/취업/skt_fly_ai/dockerhw/model/"

# 모델 로드
model = Net()
model.load_state_dict(torch.load(model_path + 'modelfile'))
model.eval()

# 이미지 전처리 함수
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    image = image.convert('RGB')
    image = transform(image).unsqueeze(0)
    return image

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # 이미지 추론
    if 'file' not in request.files:
        return render_template('index.html', error='No file uploaded.')

    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', error='No file selected.')

    try:
        image = Image.open(file)
        image = preprocess_image(image)

        with torch.no_grad():
            outputs = model(image)
            _, predicted = torch.max(outputs, 1)

        predicted_class = classes[predicted.item()]
        return render_template('index.html', prediction=predicted_class)

    except:
        return render_template('index.html', error='Error occurred during prediction.')

if __name__ == '__main__':
    app.run()
