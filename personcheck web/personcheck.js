// TensorFlow.js 모델을 로드하고 실행하는 코드
async function loadModel() {
    const model = await cocoSsd.load();
    return model;
}

// 이미지에서 객체 감지
async function detectObjects(image) {
    const model = await loadModel();
    const predictions = await model.detect(image);
    return predictions;
}

// 웹캠 스트림 가져오기
async function getWebcamStream() {
    const constraints = { video: true};
    const stream = await navigator.mediaDevices.getUserMedia(constraints);
    return stream;
}

// 웹캠 비디오 요소 생성
async function createVideoElement() {
    const video = document.createElement('video');
    const stream = await getWebcamStream();
    video.srcObject = stream;
    video.play();
    return video;
}

// 객체 감지 및 그리기
async function detectAndDrawObjects() {
    const video = await createVideoElement();
    const canvas = document.getElementById('canvas');
    const context = canvas.getContext('2d');
    const countElement = document.getElementById('count');
    const statusElement = document.getElementById('status');

    async function animate() {
        const predictions = await detectObjects(video);
        context.clearRect(0, 0, canvas.width, canvas.height);
        let personCount = 0;

        predictions.forEach(prediction => {
            if (prediction.class === 'person') {  // person만 그리도록 필터링
                const [x, y, width, height] = prediction.bbox;
                context.beginPath();
                context.rect(x, y, width, height);
                context.lineWidth = 2;
                context.strokeStyle = 'red';
                context.fillStyle = 'red';
                context.stroke();
                context.fillText(`${prediction.class} (${Math.round(prediction.score * 100)}%)`, x, y - 5);
                personCount++;
            }
        });

        countElement.textContent = `Person Count: ${personCount}`;

        // 카운트에 따라 상태 표시 업데이트
        if (personCount >= 3) {
            statusElement.textContent = '혼잡';
            statusElement.className = 'red';
        } else if (personCount === 2) {
            statusElement.textContent = '보통';
            statusElement.className = 'orange';
        } else {
            statusElement.textContent = '여유';
            statusElement.className = 'green';
        }

        requestAnimationFrame(animate);
    }

    animate();
}

// 실행
detectAndDrawObjects();
