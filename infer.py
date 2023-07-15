import torch
import torch.nn as nn
import torchvision.transforms as transforms
from flask import Flask, render_template, request
from PIL import Image


app = Flask(__name__)

# 신경망 모델 정의
class Net(nn.Module):
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

# 모델 로드
model = Net()
model.load_state_dict(torch.load('model.pth'))
model.eval()

# 입력 이미지 전처리 함수
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    return transform(image).unsqueeze(0)

# 루트 URL에 접속 시 파일 선택 폼 표시
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

# 파일 선택 후 제출 시 추론 결과 반환
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', error='No file uploaded.')

    file = request.files['file']
    image = Image.open(file)
    image = preprocess_image(image)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

    label = predicted.item()

    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
    prediction = classes[label]

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
