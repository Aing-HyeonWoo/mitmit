const webcam = document.getElementById('webcam');
        const captureBtn = document.getElementById('capture-btn');
        const capturedImage = document.getElementById('captured-image');
        const cropBtn = document.getElementById('crop-btn');
        const cancelCropBtn = document.getElementById('cancel-crop-btn');
        const croppedImage = document.getElementById('cropped-image');
        const audioUpload = document.getElementById('audio-upload');
        const audioPlayer = document.getElementById('audio-player');
        const imageUpload = document.getElementById('image-upload');
        const confirmSection = document.getElementById('confirm-section');
        const confirmBtn = document.getElementById('confirm-btn');
        const popup = document.getElementById('popup');
        const overlay = document.getElementById('overlay');
        const closePopupBtn = document.getElementById('close-popup-btn');
        let cropper, stream;
        let imageUploaded = false;
        let audioUploaded = false;

        // Start webcam
        async function startWebcam() {
            try {
                stream = await navigator.mediaDevices.getUserMedia({ video: true });
                webcam.srcObject = stream;
            } catch (error) {
                alert('Error accessing webcam: ' + error.message);
            }
        }

        // Stop webcam
        

        // Capture image from webcam
        captureBtn.addEventListener('click', () => {
            const canvas = document.createElement('canvas');
            canvas.width = webcam.videoWidth;
            canvas.height = webcam.videoHeight;
            const context = canvas.getContext('2d');
            context.drawImage(webcam, 0, 0, canvas.width, canvas.height);

            capturedImage.src = canvas.toDataURL('image/png');
            capturedImage.style.display = 'block';
            document.getElementById('cropping-section').style.display = 'block';

            // Initialize cropper
            cropper = new Cropper(capturedImage, {
                aspectRatio: 1, // Example: square crop
                viewMode: 1,
            });
        });

        // Handle crop
        cropBtn.addEventListener('click', () => {
            const canvas = cropper.getCroppedCanvas();
            croppedImage.src = canvas.toDataURL('image/png');
            document.getElementById('cropping-section').style.display = 'none';
            cropper.destroy(); // Clean up cropper instance
            imageUploaded = true;
            checkUploads();
        });

        // Cancel cropping
        cancelCropBtn.addEventListener('click', () => {
            cropper.destroy();
            document.getElementById('cropping-section').style.display = 'none';
        });

        // Stop webcam on button click

        // Start webcam on page load
        startWebcam();

        // Handle audio file upload
        audioUpload.addEventListener('change', event => {
            const file = event.target.files[0];
            if (file && file.type === 'audio/wav') {
                const reader = new FileReader();
                reader.onload = e => {
                    audioPlayer.src = e.target.result;
                    audioPlayer.style.display = 'block'; // Show the audio player
                };
                reader.readAsDataURL(file);
                audioUploaded = true;
                checkUploads();
            } else {
                alert('Please upload a WAV file.');
            }
        });

        // Handle image file upload
        imageUpload.addEventListener('change', event => {
            const file = event.target.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = e => {
                    capturedImage.src = e.target.result;
                    capturedImage.style.display = 'block';
                    document.getElementById('cropping-section').style.display = 'block';
                    cropper = new Cropper(capturedImage, {
                        aspectRatio: 1, // Example: square crop
                        viewMode: 1,
                    });
                    imageUploaded = true;
                    checkUploads();
                };
                reader.readAsDataURL(file);
            }
        });

        // Check if both image and audio are uploaded
        function checkUploads() {
            if (imageUploaded && audioUploaded) {
                confirmSection.style.display = 'block';
            }
        }

        img_class = {'buffalo':'0', 'cat':'1', 'dog':'2','elephant':'3', 'rhino':'4', 'zebra':'5'}
        audio_class = {'Cat':'0', 'Chicken':'1', 'Cow':'2', 'Dog':'3', 'Dolphin':'4', 'Frog':'5', 'Horse':'6', 'Monkey':'7', 'Sheep':'8'}

        // Open popup with API data
        confirmBtn.addEventListener('click', async () => {
            // Simulate an API call to fetch data
            const api = await fetchDataFromAPI();
            console.log(api)
            document.getElementById("yourface").innerHTML = `Your face brother :  ${api.image}..<br/> Your voice brother : ${api.audio}`
            document.getElementById('realresult').src = `animals/${img_class[api.image]}${audio_class[api.audio]}.png`;
            overlay.style.display = 'block';
            popup.style.display = 'block';
        });
        
        
        // Close popup
        closePopupBtn.addEventListener('click', () => {
            overlay.style.display = 'none';
            popup.style.display = 'none';
        });

        // Simulated API call
        async function fetchDataFromAPI() {
            const formData = new FormData();
        
            // 이미지 크롭한 이미지를 base64로 변환하여 전송 (대신 이미지 파일을 전송할 수 있음)
            const croppedCanvas = cropper.getCroppedCanvas();
            return new Promise((resolve, reject) => {
                croppedCanvas.toBlob(async function (blob) {
                    formData.append("image", blob, "cropped_image.png");
        
                    // 오디오 파일도 FormData에 추가
                    const audioFile = document.getElementById('audio-upload').files[0];
                    if (audioFile) {
                        formData.append("audio", audioFile);
                    }
        
                    try {
                        // API에 데이터 전송
                        const response = await fetch('http://localhost:5000/upload', {
                            method: 'POST',
                            body: formData,
                            mode: 'cors',
                        });
        
                        if (!response.ok) {
                            throw new Error(`HTTP error! status: ${response.status}`);
                        }
        
                        const data = await response.json();
                        console.log('Uploaded files:', data);
                        resolve(data); // 데이터를 반환
                    } catch (error) {
                        console.error('Error uploading files:', error);
                        reject(error); // 에러를 상위로 전달
                    }
                }, 'image/png'); // 이미지를 PNG 형식으로 변환하여 Blob으로 전송
            });
        }


        document.getElementById('send-message-btn').addEventListener('click', async function() {
            const userInput = document.getElementById('chat-input').value;
            if (userInput.trim() !== "") {
                // Display user's message
                const userMessageDiv = document.createElement('div');
                userMessageDiv.classList.add('user-message');
                userMessageDiv.textContent = userInput;
                document.getElementById('chat-history').appendChild(userMessageDiv);
        
                // Clear input field
                document.getElementById('chat-input').value = "";
        
                // Scroll to bottom
                document.getElementById('chat-history').scrollTop = document.getElementById('chat-history').scrollHeight;
        
                try {
                    // Fetch response from the server
                    const response = await fetch('http://localhost:5000/chat', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json' // Ensure the body is in JSON format
                        },
                        body: JSON.stringify({ q: userInput }), // Send the input data as a JSON string
                        mode: 'cors' // Ensure CORS is enabled if needed
                    });
        
                    if (!response.ok) {
                        throw new Error(`HTTP error! status: ${response.status}`);
                    }
        
                    // Parse JSON response from server
                    const data = await response.json();
                    console.log(data); // For debugging, check the response
        
                    // Display bot's response
                    setTimeout(() => {
                        const botMessageDiv = document.createElement('div');
                        botMessageDiv.classList.add('bot-message');
                        botMessageDiv.textContent = data.a || "No response from bot"; // Assuming response field from bot's JSON
                        document.getElementById('chat-history').appendChild(botMessageDiv);
                        document.getElementById('chat-history').scrollTop = document.getElementById('chat-history').scrollHeight;
                    }, 1000);
        
                } catch (error) {
                    console.error('Error:', error);
                    // Handle error (e.g., display error message in chat)
                    setTimeout(() => {
                        const botMessageDiv = document.createElement('div');
                        botMessageDiv.classList.add('bot-message');
                        botMessageDiv.textContent = "Sorry, there was an error processing your request.";
                        document.getElementById('chat-history').appendChild(botMessageDiv);
                        document.getElementById('chat-history').scrollTop = document.getElementById('chat-history').scrollHeight;
                    }, 1000);
                }
            }
        });
        
        
        
