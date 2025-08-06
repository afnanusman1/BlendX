from django.shortcuts import render, HttpResponse, redirect
from django.contrib.auth import authenticate, login
from django.db.models import Q
from django.core.mail import send_mail
from django.http import FileResponse, JsonResponse
from datetime import datetime as dt
from tensorflow.keras.models import load_model
from django.core.files.base import ContentFile
from io import BytesIO
import io
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
import hashlib
from PIL import Image
import numpy as np
from gtts import gTTS
import os
import base64
from gtts import gTTS
from .models import *
import datetime
import tensorflow as tf
# Updated imports
from django.views.decorators.csrf import csrf_protect
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
from Crypto.Random import get_random_bytes
import hashlib
import hmac
import binascii
from django.core.cache import cache
from django.conf import settings
import logging
from transformers import pipeline
import tempfile
# Create your views here.

speech_recognizer = pipeline("automatic-speech-recognition", model="openai/whisper-small")

def transcribe_audio(request):
    if request.method == 'POST' and request.FILES.get('audio'):
        audio_file = request.FILES['audio']
        
        # Save temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
            for chunk in audio_file.chunks():
                tmp.write(chunk)
            tmp.seek(0)
            
            # Perform speech recognition
            result = speech_recognizer(tmp.name)
            
        return JsonResponse({'transcription': result['text']})
    
    return JsonResponse({'error': 'Invalid request'}, status=400)
def index(request):
    return render(request, "index.html")

def contact(request):
    return render(request, "contact.html")

def signin(request):
    if request.POST:
        email = request.POST["email"]
        passw = request.POST["password"]
        data = authenticate(username=email, password=passw)
        if data is not None:
            login(request, data)
            print("Data")
            if data.is_active:
                if data.userType == "User":
                    print("User")
                    id = data.id
                    request.session["uid"] = id
                    resp = '<script>alert("Login Success"); window.location.href = "/userHome";</script>'
                    return HttpResponse(resp)
                elif data.userType == "Admin":
                    print("Admin")
                    resp = '<script>alert("Login Success"); window.location.href = "/adminHome";</script>'
                    return HttpResponse(resp)
            else:
                print("Sorry You Are Not Approved")
                resp = '<script>alert("Sorry You Are Not Approved"); window.location.href = "/adminHome";</script>'
                return HttpResponse(resp)
        else:
            resp = '<script>alert("Sorry You Are Not Approved..😥");window.location.href="/login"</script>'
            return HttpResponse(resp)
    return render(request, "COMMON/login.html")

def register(request):
    current_datetime = datetime.date.today()
    current_date = current_datetime.strftime("%Y-%m-%d")
    print(current_date)
    if request.POST:
        name = request.POST["name"]
        email = request.POST["email"]
        gender = request.POST["gender"]
        dob = request.POST["dob"]
        phone = request.POST["phone"]
        password = request.POST["password"]
        address = request.POST["address"]
        image = request.FILES["imgfile"]

        if Login.objects.filter(username=email).exists():
            return HttpResponse(
                "<script>alert('Email already Exists Added');window.location.href='/registration'</script>"
            )
        else:
            logQry = Login.objects.create_user(
                username=email,
                password=password,
                userType="User",
                viewPass=password,
                is_active=0,
            )
            logQry.save()
            regQry = Person.objects.create(
                name=name,
                email=email,
                gender=gender,
                dob=dob,
                phone=phone,
                address=address,
                image=image,
                loginid=logQry,
            )
            regQry.save()
            return HttpResponse(
                "<script>alert('Registration Successfull');window.location.href='/login'</script>"
            )
    return render(request, "COMMON/register.html", {"current_date": current_date})

def udp(request):
    abc = Login.objects.get(username="admin@gmail.com")
    abc.set_password("admin")
    abc.save()
    return HttpResponse("Success")

################################################***ADMIN***################################################

def adminHome(request):
    return render(request, "ADMIN/adminHome.html")

def viewUsers(request):
    data = Person.objects.all()
    return render(request, "ADMIN/viewUsers.html", {"data": data})

def approveUser(request):
    id = request.GET["id"]
    approve = Login.objects.filter(id=id).update(is_active=1)
    return HttpResponse(
        "<script>alert('Approved');window.location.href='/viewusers'</script>"
    )

def rejectUser(request):
    id = request.GET["id"]
    approve = Login.objects.filter(id=id).update(is_active=0)
    return HttpResponse(
        "<script>alert('Rejected');window.location.href='/viewusers'</script>"
    )

def deleteUser(request):
    id = request.GET["id"]
    delete = Login.objects.filter(id=id).delete()
    return HttpResponse(
        "<script>alert('Deleted');window.location.href='/viewusers'</script>"
    )

def approveRequest(request):
    id = request.GET["id"]
    upDate = Person.objects.filter(loginid=id).update(status="Approved")
    return HttpResponse(
        "<script>alert('Approved');window.location.href='/viewusers'</script>"
    )

def viewFeedback(request):
    data = Feedback.objects.all()
    return render(request, "ADMIN/viewFeedback.html", {"data": data})

################################################***USER***################################################

def userHome(request):
    return render(request, "USER/userHome.html")

def profile(request):
    uid = request.session["uid"]
    data = Person.objects.get(loginid=uid)
    return render(request, "USER/profile.html", {"data": data})

def requestUpdation(request):
    id = request.GET["id"]
    requestUpdate = Person.objects.filter(loginid=id).update(status="Requested")
    return HttpResponse(
        "<script>alert('Requested');window.location.href='/profile'</script>"
    )

def updateProfile(request):
    current_datetime = datetime.date.today()
    current_date = current_datetime.strftime("%Y-%m-%d")
    print(current_date)
    uid = request.session["uid"]
    data = Person.objects.get(loginid=uid)
    if request.POST:
        name = request.POST["name"]
        email = request.POST["email"]
        gender = request.POST["gender"]
        dob = request.POST["dob"]
        phone = request.POST["phone"]
        password = request.POST["password"]
        address = request.POST["address"]
        image = request.FILES.get("imgfile")

        if image:
            data = Person.objects.get(loginid=uid)
            data.image = image
            data.save()
        if password:
            data = Login.objects.get(id=uid)
            data.set_password(password)
            data.save()
        update = Person.objects.filter(loginid=uid).update(
            name=name,
            email=email,
            phone=phone,
            address=address,
            gender=gender,
            dob=dob,
            status="View",
        )
        logUpdate = Login.objects.filter(id=uid).update(username=email)
        return redirect("/profile")
    return render(
        request, "USER/updateProfile.html", {"data": data, "current_date": current_date}
    )

def addFeedback(request):
    uid = request.session["uid"]
    UID = Person.objects.get(loginid=uid)
    data = Feedback.objects.filter(uid__loginid=uid)
    if request.POST:
        title = request.POST["title"]
        feedback = request.POST["feedback"]
        add = Feedback.objects.create(uid=UID, title=title, feedback=feedback)
        add.save()
        return HttpResponse(
            "<script>alert('Feedback Added');window.location.href='/addFeedback'</script>"
        )
    return render(request, "USER/addFeedback.html", {"data": data})



# Reversible Data Hiding (RDH) Implementation
# Updated Reversible Data Hiding (RDH) Implementation
class ReversibleDataHiding:
    @staticmethod
    def embed_data(image, data):
        """
        Embed data while storing original LSBs for reversibility.
        """
        try:
            img_array = np.array(image)
            height, width, channels = img_array.shape

            # Convert data to binary with header
            if isinstance(data, str):
                data = data.encode('utf-8')
            data_length = len(data).to_bytes(4, 'big')
            checksum = sum(data).to_bytes(4, 'big')
            payload = data_length + checksum + data

            # Collect original LSBs
            total_bits = height * width * channels
            original_lsbs = [
                str(img_array[i, j, k] & 1)
                for i in range(height)
                for j in range(width)
                for k in range(channels)
            ][: len(payload) * 8]  # Only store LSBs overwritten by payload

            # Append original LSBs to payload for recovery
            payload += bytes([int(''.join(original_lsbs[i:i+8]), 2) for i in range(0, len(original_lsbs), 8)])

            # Convert payload to binary
            data_bin = ''.join(format(byte, '08b') for byte in payload)
            data_len = len(data_bin)

            # Embed data into LSBs
            data_index = 0
            for i in range(height):
                for j in range(width):
                    for k in range(channels):
                        if data_index < data_len:
                            bit = int(data_bin[data_index])
                            img_array[i, j, k] = (img_array[i, j, k] & 0xFE) | bit
                            data_index += 1
                        else:
                            img_array[i, j, k] &= 0xFE  # Clear remaining LSBs

            return Image.fromarray(img_array)
        except Exception as e:
            raise Exception(f"Embedding failed: {str(e)}")

    @staticmethod
    def extract_data(image):
        """
        Extract data and restore original LSBs.
        """
        try:
            img_array = np.array(image.convert('RGB'))
            height, width, channels = img_array.shape

            # Extract all LSBs
            data_bin = ''.join(str(img_array[i, j, k] & 1) for i in range(height) for j in range(width) for k in range(channels))

            # Convert to bytes
            data_bytes = bytearray()
            for i in range(0, len(data_bin), 8):
                byte_str = data_bin[i:i+8]
                if len(byte_str) < 8:
                    break
                data_bytes.append(int(byte_str, 2))

            # Parse header and original LSBs
            length = int.from_bytes(data_bytes[:4], 'big')
            stored_checksum = int.from_bytes(data_bytes[4:8], 'big')
            extracted_data = data_bytes[8 : 8 + length]
            original_lsbs = data_bytes[8 + length :]

            # Validate checksum
            if sum(extracted_data) != stored_checksum:
                raise ValueError("Checksum mismatch")

            # Restore original LSBs
            data_index = 0
            original_lsbs_bin = ''.join(format(byte, '08b') for byte in original_lsbs)
            for i in range(height):
                for j in range(width):
                    for k in range(channels):
                        if data_index < len(original_lsbs_bin):
                            original_bit = int(original_lsbs_bin[data_index])
                            img_array[i, j, k] = (img_array[i, j, k] & 0xFE) | original_bit
                            data_index += 1
                        else:
                            break

            # Return both extracted message and original image
            restored_image = Image.fromarray(img_array)
            return extracted_data.decode('utf-8'), restored_image
        except Exception as e:
            raise Exception(f"Extraction failed: {str(e)}")
#class ReversibleDataHiding:
    @staticmethod
    def embed_data1(image, data):
        """
        Embed data into an image using reversible data hiding with header.
        :param image: PIL Image object
        :param data: Data to embed (string or bytes)
        :return: PIL Image object with embedded data
        """
        try:
            # Convert PIL Image to NumPy array and ensure RGB
            if image.mode != 'RGB':
                image = image.convert('RGB')
            img_array = np.array(image)
            height, width, channels = img_array.shape
            if isinstance(data, str):
                data = data.encode('utf-8')
            
            # Create header (4 bytes length + 4 bytes checksum)
            data_length = len(data)
            length_bytes = data_length.to_bytes(4, byteorder='big')
            checksum = sum(data)
            checksum_bytes = checksum.to_bytes(4, byteorder='big')
            data_to_hide = length_bytes + checksum_bytes + data
            
            # Convert data to binary
            data_bin = ''.join(format(byte, '08b') for byte in data_to_hide)
            data_len = len(data_bin)
            
            # Check image capacity
            height, width, channels = img_array.shape
            if data_len > height * width * channels:
                raise ValueError("Data too large for the image")
            
            # Embed data into LSBs
            data_index = 0
            for i in range(height):
                for j in range(width):
                    for k in range(channels):
                        if data_index < data_len:
                            bit = int(data_bin[data_index])
                            img_array[i, j, k] = (img_array[i, j, k] & 0xFE) | bit
                            data_index += 1
                        else:
                            # Fill remaining pixels with 0 bits
                            img_array[i, j, k] = img_array[i, j, k] & 0xFE
            
            return Image.fromarray(img_array)
        except Exception as e:
            raise Exception(f"Reversible Data Hiding error: {str(e)}")
    @staticmethod
    def extract_data1(image):
        try:
            # Convert PIL Image to NumPy array and ensure RGB
            if image.mode != 'RGB':
                image = image.convert('RGB')
            img_array = np.array(image)
            height, width, channels = img_array.shape
            # Extract LSBs
            data_bin = ''.join(str(img_array[i, j, k] & 1) for i in range(height) for j in range(width) for k in range(channels))

            # Convert binary to bytes
            data_bytes = bytearray()
            for i in range(0, len(data_bin), 8):
                byte_str = data_bin[i:i+8]
                if len(byte_str) < 8:
                    break
                data_bytes.append(int(byte_str, 2))

            # Validate header
            if len(data_bytes) < 8:
                raise ValueError("No valid header found")

            # Parse header
            length = int.from_bytes(data_bytes[:4], byteorder='big')
            stored_checksum = int.from_bytes(data_bytes[4:8], byteorder='big')
            extracted_data = bytes(data_bytes[8 : 8 + length])

            # Validate checksum
            actual_checksum = sum(extracted_data)
            if actual_checksum != stored_checksum:
                raise ValueError("Data corrupted: Checksum mismatch")

            return extracted_data.decode('utf-8')
        except Exception as e:
            raise Exception(f"Extraction failed: {str(e)}")
            
# Image Preprocessing Steps
class ImagePreprocessor:
    @staticmethod
    def preprocess_image(image):
        """
        Preprocess the image for better data hiding.
        :param image: PIL Image object
        :return: Preprocessed PIL Image object
        """
        try:
            ## Convert PIL Image to OpenCV format
            #img_array = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            ## Apply Gaussian blur for noise reduction
            #img_array = cv2.GaussianBlur(img_array, (5, 5), 0)
            
            ## Apply histogram equalization for better contrast
            #img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2YCrCb)
            #channels = cv2.split(img_array)
            #cv2.equalizeHist(channels[0], channels[0])
            #img_array = cv2.merge(channels)
            #img_array = cv2.cvtColor(img_array, cv2.COLOR_YCrCb2BGR)
            
            ## Convert back to PIL Image
            #return Image.fromarray(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))
            if image.mode != 'RGB':
                image = image.convert('RGB')
            return image
        except Exception as e:
            raise Exception(f"Image preprocessing error: {str(e)}")


class ImageTamperDetector:
    def __init__(self, model_path):
        self.model = load_model(model_path)

    def detect_tampering(self, image):
        """Detect if the image has been tampered with"""
        # Preprocess the image
        processed_image = self.preprocess_image(image)
        # Predict using the model
        prediction = self.model.predict(processed_image)
        return prediction

    def preprocess_image(self, image):
        """Preprocess the image for the model"""
        # Implement preprocessing steps here
        pass    

# Updated compose function to use RDH
def compose(request):
    uid = request.session["uid"]
    sender = Person.objects.get(loginid=uid)
    print(sender.email)
    msg = ""
    if "sent" in request.POST:
        # ... [existing code up to encryption] ...
        receiver = request.POST["email"]
        subject = request.POST["subject"]
        password = request.POST["password"]
        message = request.POST["message"]
        image = request.FILES["imgfile"]
        attachment = request.FILES.get("attachment")
        print(receiver)
        emailSubject = "Password for Opening the File"
        body = f"The password for opening the specific file is: {password}"
        
        if Person.objects.filter(email=receiver).exists():
            # Preprocess the image
            original_format = Image.open(attachment).format
            cover_image = Image.open(attachment).convert('RGB')
            preprocessed_image = ImagePreprocessor.preprocess_image(cover_image)
            # Embed the message using RDH
            stego_image = ReversibleDataHiding.embed_data(preprocessed_image, message)
            
            # Save stego image to bytes
            img_byte_arr = BytesIO()
            # Preserve original format if supported (use PNG for lossless)
            if original_format in ['JPEG', 'PNG', 'BMP', 'TIFF', 'WEBP','JPG']:  # Lossless formats
                stego_image.save(img_byte_arr, format=original_format)
            else:  # Default to PNG for JPEGs or unknown formats
                stego_image.save(img_byte_arr, format='PNG')
            stego_bytes = img_byte_arr.getvalue()  # Get the bytes
            # Encrypt the stego image
            key = hashlib.sha256(password.encode()).digest()
            cipher = AES.new(key, AES.MODE_ECB)
            encrypted_img = cipher.encrypt(pad(stego_bytes, AES.block_size))
            print("the encryption cipher before:", cipher)           
            # Create the Message object
            receiverId = Person.objects.get(email=receiver)
            recipient_list = [receiver]
            result = send_mail(
                subject=emailSubject,
                message=body,
                from_email=sender.email,
                recipient_list=recipient_list,
            )
            sentMsg = Message(
                sender=sender,
                receiver=receiverId,
                subject=subject,
                msg= "",#message,
                password=password,
                file=ContentFile(encrypted_img, name=f"encry_{image.name.split('.')[0]}.png"),  # Save encrypted data directly
                attachment="",#attachment,
                status="Sent",
                encryption_key="",
                processed_image=image,
                audio_file=None
            )
            sentMsg.save()  # This will save the file to the database/storage
            
            return HttpResponse(
                "<script>alert('Message Sent Successfully');window.location.href='/compose'</script>"
            )
        else:
            msg = "User does not exist"
            return render(request, "USER/compose.html", {"msg": msg})
    return render(request, "USER/compose.html")

def inbox(request):
    uid = request.session["uid"]
    inboxData = Message.objects.filter(receiver__loginid=uid).order_by("-date")
    print(inboxData)
    return render(request, "USER/inbox.html", {"inboxData": inboxData})


# Updated readMail function to use RDH
def readMail(request):
    uid = request.session["uid"]
    id = request.GET["id"]
    data = Message.objects.get(id=id)

    if request.POST:
        password = request.POST["password"]
        #try:
        if Message.objects.filter(Q(id=id) & Q(password=password)).exists():
            print("mailData exist")
            mailData = Message.objects.get(Q(id=id) & Q(password=password))
            #mailData = Message.objects.get(id=id, password=password)
            mailData.file.seek(0)
            encrypted_img = mailData.file.read()

            #if not encrypted_img:
            #   return HttpResponse(
            #      f"<script>alert('File is empty or corrupted.');window.location.href='/readMail?id={id}'</script>"
            # )

            # Decrypt the image
            key = hashlib.sha256(password.encode()).digest()
            cipher = AES.new(key, AES.MODE_ECB)
            decrypted_img = cipher.decrypt(encrypted_img)
            print("the decryption cipher after:", cipher)
            try:
                decrypted_img = unpad(decrypted_img, AES.block_size)
            except ValueError as e:
                return HttpResponse(
                    f"<script>alert('Decryption error: Incorrect padding.');window.location.href='/readMail?id={id}'</script>"
                )
            # Validate decrypted image
            try:
                stego_image = Image.open(BytesIO(decrypted_img)).convert('RGB')
                stego_image.verify()  # Verify integrity
                stego_image = Image.open(BytesIO(decrypted_img))  # Reopen as verify closes the image
                #stego_image = Image.open(BytesIO(decrypted_img)).convert('RGB') # Preserve transparency if needed
                # Save as PNG if required
            except Exception as e:
                return HttpResponse(f"<script>alert('Invalid image: {str(e)}');window.location.href='/readMail?id={id}'</script>")
            # Extract the hidden message
            stego_image.save("output.png", format="PNG")
            extracted_message = ReversibleDataHiding.extract_data(stego_image)
            # Save the stego image to bytes in PNG format to serve it
            img_byte_arr = BytesIO()
            stego_image.save(img_byte_arr, format='PNG')
            img_bytes = img_byte_arr.getvalue()
            img_base64 = base64.b64encode(img_bytes).decode('utf-8')  # Encode to base64
            img_data_url = f"data:image/png;base64,{img_base64}"  # Data URL
            
            # Generate audio if requested
            #audio_path = None
            #if "generate_audio" in request.POST:
            #    audio_path = AudioProcessor.text_to_audio(
            #       extracted_message,
            #        f'./static/downloads/output_{dt.now().strftime("%Y%m%d_%H%M%S")}.mp3'
            #   )
            #   mailData.audio_file = audio_path  # Save the audio file path
            #   mailData.save()
            print("the extracted message:", extracted_message[0])
            if mailData:
                return render(
                    request,
                    "USER/readMail.html",
                    {"image_data_url": img_data_url, "mailData": mailData,"encryption_key": extracted_message[0]},
                )
            #context = {
            #    "data": stego_image,
            #    "mailData": mailData,
            #    "decrypted_message": extracted_message
            #}
            
            # ... [rest of the code for audio generation] ...
            return render(request, "USER/readMail.html", {"data": data, "extracted_message": extracted_message[0]})
        else:
            print("Else")
            return HttpResponse(
                f"<script>alert('Incorrect Password');window.location.href='/readMail?id={id}'</script>"
            )
        #except Message.DoesNotExist:
            print("No message exists")
            return HttpResponse(
                f"<script>alert('Incorrect Password');window.location.href='/readMail?id={id}'</script>"
            )
        #except (ValueError, Exception) as e:
            print(f"Error: {str(e)}")
            return HttpResponse(
                f"<script>alert('Error: {str(e)}');window.location.href='/readMail?id={id}'</script>"
            )
    return render(request, "USER/readMail.html", {"data": data.file})

def text_to_audio(text, output_file="output.mp3", language="en"):
    try:
        tts = gTTS(text=text, lang=language)
        tts.save(output_file)
        print(f"Text converted to audio and saved as '{output_file}'")
        os.system(f"start {output_file}")
    except Exception as e:
        print(f"An error occurred: {e}")
def download(request):
    current_datetime = dt.now().strftime("%Y-%m-%d_%H-%M-%S")
    text = request.GET["text"]
    id = request.GET["id"]
    print(text)
    output_file_path = f"./static/downloads/output_{current_datetime}_{id}.mp3"
    text_to_audio(text, output_file_path)
    return redirect(f"/readMail?id={id}")


#def download(request):
#   mailData = Message.objects.get(id=id)
#    if mailData.audio_file:
#        file_path = mailData.audio_file.path
#        if os.path.exists(file_path):
#            with open(file_path, 'rb') as fh:
#                response = HttpResponse(fh.read(), content_type="audio/mpeg")
#               response['Content-Disposition'] = f'attachment; filename={os.path.basename(file_path)}'
#                return response
#    return HttpResponse(
#        "<script>alert('File not found');window.location.href='/readMail?id={id}'</script>"
#    )

def send_email(request):
    if request.POST:
        sub = request.POST["subject"]
        msg = request.POST["message"]
        receiver = request.POST["receiver"]
        from_email = "gauthamts484@gmail.com"
        recipient_list = [receiver]
        result = send_mail(sub, msg, from_email, recipient_list)
        print(result)
    return render(request, "sendMail.html")
