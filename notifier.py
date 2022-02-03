import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from cv2 import cv2

class Notifier:

    def __init__(self, sender_address, physical_address, receiver_address, password):
        self.sender_address = sender_address
        self.physical_address = physical_address
        self.receiver_address = receiver_address
        self.password = password

    def notify(self, image):
        message = MIMEMultipart()
        message['From'] = self.sender_address
        message['To'] = self.receiver_address
        message['Subject'] = 'EMERGENCY'
        content = 'Possible violent event at ' + self.physical_address 
        content = content + ". Possible attacker shown in image"
        message.attach(MIMEText(content, 'plain'))
        cv2.imwrite('WARNING.jpg', image)
        attach_file_name = 'WARNING.jpg'
        with open(attach_file_name, 'rb') as f:
            img_data = f.read()
        img = MIMEImage(img_data, name=attach_file_name)
        message.attach(img)
        session = smtplib.SMTP('smtp.gmail.com', 587)
        session.starttls()
        session.login(self.sender_address, self.password)
        text = message.as_string()
        session.sendmail(self.sender_address, self.receiver_address, text)
        session.quit()
        print("Notified authorities")
