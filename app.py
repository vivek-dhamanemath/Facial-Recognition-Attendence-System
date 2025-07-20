import cv2
import os
from flask import Flask, request, render_template, send_file, make_response
from datetime import date
from datetime import datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch

# Defining Flask App
app = Flask(__name__)

nimgs = 10

# Saving Date today in 2 different formats
datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")


# Initializing VideoCapture object to access WebCam
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


# If these directories don't exist, create them
if not os.path.isdir('Attendance'):
    os.makedirs('Attendance')
if not os.path.isdir('static'):
    os.makedirs('static')
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')
if f'Attendance-{datetoday}.csv' not in os.listdir('Attendance'):
    with open(f'Attendance/Attendance-{datetoday}.csv', 'w') as f:
        f.write('Name,Roll,Time')


# get a number of total registered users
def totalreg():
    return len(os.listdir('static/faces'))


# extract the face from an image
def extract_faces(img):
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_points = face_detector.detectMultiScale(gray, 1.2, 5, minSize=(20, 20))
        return face_points
    except:
        return []


# Identify face using ML model
def identify_face(facearray):
    model = joblib.load('static/face_recognition_model.pkl')
    return model.predict(facearray)


# A function which trains the model on all the faces available in faces folder
def train_model():
    faces = []
    labels = []
    userlist = os.listdir('static/faces')
    for user in userlist:
        for imgname in os.listdir(f'static/faces/{user}'):
            img = cv2.imread(f'static/faces/{user}/{imgname}')
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face.ravel())
            labels.append(user)
    faces = np.array(faces)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces, labels)
    joblib.dump(knn, 'static/face_recognition_model.pkl')


# Extract info from today's attendance file in attendance folder
def extract_attendance():
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    names = df['Name']
    rolls = df['Roll']
    times = df['Time']
    l = len(df)
    return names, rolls, times, l


# Add Attendance of a specific user
def add_attendance(name):
    username = name.split('_')[0]
    userid = name.split('_')[1]
    current_time = datetime.now().strftime("%H:%M:%S")

    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    if int(userid) not in list(df['Roll']):
        with open(f'Attendance/Attendance-{datetoday}.csv', 'a') as f:
            f.write(f'\n{username},{userid},{current_time}')


## A function to get names and rol numbers of all users
def getallusers():
    userlist = os.listdir('static/faces')
    names = []
    rolls = []
    l = len(userlist)

    for i in userlist:
        name, roll = i.split('_')
        names.append(name)
        rolls.append(roll)

    return userlist, names, rolls, l


## A function to delete a user folder 
def deletefolder(duser):
    pics = os.listdir(duser)
    for i in pics:
        os.remove(duser+'/'+i)
    os.rmdir(duser)




################## ROUTING FUNCTIONS #########################

# Landing page route (now the main homepage)
@app.route('/')
def landing():
    return render_template('landing.html')

# Dashboard page
@app.route('/dashboard')
def home():
    names, rolls, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)


## List users page
@app.route('/listusers')
def listusers():
    userlist, names, rolls, l = getallusers()
    return render_template('listusers.html', userlist=userlist, names=names, rolls=rolls, l=l, totalreg=totalreg(), datetoday2=datetoday2)


## Delete functionality
@app.route('/deleteuser', methods=['GET'])
def deleteuser():
    duser = request.args.get('user')
    deletefolder('static/faces/'+duser)

    ## if all the face are deleted, delete the trained file...
    if os.listdir('static/faces/')==[]:
        os.remove('static/face_recognition_model.pkl')
    
    try:
        train_model()
    except:
        pass

    userlist, names, rolls, l = getallusers()
    return render_template('listusers.html', userlist=userlist, names=names, rolls=rolls, l=l, totalreg=totalreg(), datetoday2=datetoday2)


## Export CSV functionality
@app.route('/export-csv')
@app.route('/export_csv')
def export_csv():
    try:
        # Get the same data that's displayed on the dashboard
        names, rolls, times, l = extract_attendance()
        
        print(f"Export CSV: Found {l} attendance records")
        print(f"Names: {names}")
        print(f"Rolls: {rolls}")
        print(f"Times: {times}")
        
        # Create CSV content
        csv_content = "Name,Roll,Time\n"
        
        if l > 0:
            for i in range(l):
                csv_content += f"{names[i]},{rolls[i]},{times[i]}\n"
        
        # Create response
        response = make_response(csv_content)
        response.headers['Content-Type'] = 'text/csv'
        response.headers['Content-Disposition'] = f'attachment; filename=Attendance-{datetoday2}.csv'
        
        print("CSV file created successfully")
        return response
        
    except Exception as e:
        print(f"Error in export_csv: {str(e)}")
        import traceback
        traceback.print_exc()
## Simple Export functionality that works directly with file
@app.route('/simple-csv')
def simple_csv():
    try:
        # Read the attendance file directly
        csv_file_path = f'Attendance/Attendance-{datetoday}.csv'
        
        if os.path.exists(csv_file_path):
            with open(csv_file_path, 'r') as file:
                csv_content = file.read()
        else:
            csv_content = "Name,Roll,Time\n"
        
        response = make_response(csv_content)
        response.headers['Content-Type'] = 'text/csv'
        response.headers['Content-Disposition'] = f'attachment; filename=Attendance-{datetoday2}.csv'
        
        return response
    except Exception as e:
        return f"Error: {str(e)}", 500


@app.route('/simple-excel')
def simple_excel():
    try:
        import tempfile
        
        # Read the attendance file directly
        csv_file_path = f'Attendance/Attendance-{datetoday}.csv'
        
        if os.path.exists(csv_file_path):
            df = pd.read_csv(csv_file_path)
        else:
            df = pd.DataFrame(columns=['Name', 'Roll', 'Time'])
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp:
            df.to_excel(tmp.name, index=False)
            tmp_path = tmp.name
        
        # Send the file
        return send_file(
            tmp_path,
            as_attachment=True,
            download_name=f'Attendance-{datetoday2}.xlsx',
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
    except Exception as e:
        return f"Error: {str(e)}", 500


@app.route('/simple-pdf')
def simple_pdf():
    try:
        import tempfile
        
        # Read the attendance file directly
        csv_file_path = f'Attendance/Attendance-{datetoday}.csv'
        
        if os.path.exists(csv_file_path):
            df = pd.read_csv(csv_file_path)
        else:
            df = pd.DataFrame(columns=['Name', 'Roll', 'Time'])
        
        # Create PDF in memory
        import io
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        
        # Get styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=18,
            spaceAfter=30,
            alignment=1,  # Center alignment
            textColor=colors.HexColor('#2E75B6')
        )
        
        # Create content
        content = []
        
        # Add title
        title = Paragraph(f"Attendance Report - {datetoday2}", title_style)
        content.append(title)
        content.append(Spacer(1, 12))
        
        if len(df) > 0:
            # Prepare data for table
            data = [['S.No.', 'Name', 'ID', 'Time']]  # Header
            for i, row in df.iterrows():
                data.append([str(i+1), str(row['Name']), str(row['Roll']), str(row['Time'])])
            
            # Create table
            table = Table(data)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2E75B6')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 1), (-1, -1), 10),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey])
            ]))
            
            content.append(table)
        else:
            # Add message for no data
            no_data = Paragraph("No attendance records found for today.", styles['Normal'])
            content.append(no_data)
        
        # Add footer
        content.append(Spacer(1, 20))
        footer = Paragraph(f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Attend-AI System", styles['Normal'])
        content.append(footer)
        
        # Build PDF
        doc.build(content)
        buffer.seek(0)
        
        # Create response
        response = make_response(buffer.read())
        response.headers['Content-Type'] = 'application/pdf'
        response.headers['Content-Disposition'] = f'attachment; filename=Attendance-{datetoday2}.pdf'
        
        return response
    except Exception as e:
        return f"Error: {str(e)}", 500


## Export Excel functionality
@app.route('/export-excel')
@app.route('/export_excel')
def export_excel():
    try:
        import io
        
        # Get the same data that's displayed on the dashboard
        names, rolls, times, l = extract_attendance()
        
        print(f"Export Excel: Found {l} attendance records")
        print(f"Names: {names}")
        print(f"Rolls: {rolls}")
        print(f"Times: {times}")
        
        # Create DataFrame from the actual dashboard data
        if l > 0:
            df = pd.DataFrame({
                'Name': names,
                'Roll': rolls,
                'Time': times
            })
        else:
            df = pd.DataFrame(columns=['Name', 'Roll', 'Time'])
        
        # Create Excel file in memory
        output = io.BytesIO()
        
        # Create Excel writer with formatting
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # Write data to Excel
            df.to_excel(writer, sheet_name='Attendance', index=False)
            
            # Get the workbook and worksheet
            workbook = writer.book
            worksheet = writer.sheets['Attendance']
            
            # Add header formatting
            try:
                from openpyxl.styles import Font, PatternFill, Alignment
                
                header_font = Font(bold=True, color='FFFFFF')
                header_fill = PatternFill(start_color='2E75B6', end_color='2E75B6', fill_type='solid')
                
                # Format header row
                for cell in worksheet[1]:
                    cell.font = header_font
                    cell.fill = header_fill
                    cell.alignment = Alignment(horizontal='center')
                
                # Auto-adjust column widths
                for column in worksheet.columns:
                    max_length = 0
                    column_letter = column[0].column_letter
                    for cell in column:
                        try:
                            if len(str(cell.value)) > max_length:
                                max_length = len(str(cell.value))
                        except:
                            pass
                    adjusted_width = min(max_length + 2, 50)
                    worksheet.column_dimensions[column_letter].width = adjusted_width
            except ImportError:
                print("Styling not available, creating basic Excel file")
                pass
        
        output.seek(0)
        
        # Create response
        response = make_response(output.read())
        response.headers['Content-Type'] = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        response.headers['Content-Disposition'] = f'attachment; filename=Attendance-{datetoday2}.xlsx'
        
        print("Excel file created successfully")
        return response
        
    except Exception as e:
        print(f"Error in export_excel: {str(e)}")
        import traceback
        traceback.print_exc()
        return f"Error exporting Excel: {str(e)}", 500


## Export PDF functionality
@app.route('/export-pdf')
@app.route('/export_pdf')
def export_pdf():
    try:
        import io
        
        # Get the same data that's displayed on the dashboard
        names, rolls, times, l = extract_attendance()
        
        print(f"Export PDF: Found {l} attendance records")
        print(f"Names: {names}")
        print(f"Rolls: {rolls}")
        print(f"Times: {times}")
        
        # Create DataFrame from the actual dashboard data
        if l > 0:
            df = pd.DataFrame({
                'Name': names,
                'Roll': rolls,
                'Time': times
            })
        else:
            df = pd.DataFrame(columns=['Name', 'Roll', 'Time'])
        
        # Create PDF in memory
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        
        # Get styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=18,
            spaceAfter=30,
            alignment=1,  # Center alignment
            textColor=colors.HexColor('#2E75B6')
        )
        
        # Create content
        content = []
        
        # Add title
        title = Paragraph(f"Attendance Report - {datetoday2}", title_style)
        content.append(title)
        content.append(Spacer(1, 12))
        
        if len(df) > 0:
            # Prepare data for table
            data = [['S.No.', 'Name', 'ID', 'Time']]  # Header
            for i, row in df.iterrows():
                data.append([str(i+1), str(row['Name']), str(row['Roll']), str(row['Time'])])
            
            # Create table
            table = Table(data)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2E75B6')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 1), (-1, -1), 10),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey])
            ]))
            
            content.append(table)
        else:
            # Add message for no data
            no_data = Paragraph("No attendance records found for today.", styles['Normal'])
            content.append(no_data)
        
        # Add footer
        content.append(Spacer(1, 20))
        footer = Paragraph(f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Attend-AI System", styles['Normal'])
        content.append(footer)
        
        # Build PDF
        doc.build(content)
        buffer.seek(0)
        
        # Create response
        response = make_response(buffer.read())
        response.headers['Content-Type'] = 'application/pdf'
        response.headers['Content-Disposition'] = f'attachment; filename=Attendance-{datetoday2}.pdf'
        
        print("PDF file created successfully")
        return response
        
    except Exception as e:
        print(f"Error in export_pdf: {str(e)}")
        import traceback
        traceback.print_exc()
        return f"Error exporting PDF: {str(e)}", 500


# Test route to verify routing is working
@app.route('/test')
def test_route():
    return "Route is working! CSV export should work too."


# Debug route to check attendance data
@app.route('/debug-attendance')
def debug_attendance():
    try:
        names, rolls, times, l = extract_attendance()
        return f"""
        <h2>Debug Attendance Data</h2>
        <p><strong>Records found:</strong> {l}</p>
        <p><strong>Names:</strong> {names}</p>
        <p><strong>Rolls:</strong> {rolls}</p>
        <p><strong>Times:</strong> {times}</p>
        <p><strong>Date today:</strong> {datetoday}</p>
        <p><strong>Date today2:</strong> {datetoday2}</p>
        <hr>
        <a href="/export-csv">Test CSV Export</a> | 
        <a href="/export-excel">Test Excel Export</a> | 
        <a href="/export-pdf">Test PDF Export</a>
        """
    except Exception as e:
        return f"Error: {str(e)}"


# Our main Face Recognition functionality. 
# This function will run when we click on Take Attendance Button.
@app.route('/start', methods=['GET'])
def start():
    names, rolls, times, l = extract_attendance()

    if 'face_recognition_model.pkl' not in os.listdir('static'):
        return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2, mess='There is no trained model in the static folder. Please add a new face to continue.')

    ret = True
    cap = cv2.VideoCapture(0)
    while ret:
        ret, frame = cap.read()
        if len(extract_faces(frame)) > 0:
            (x, y, w, h) = extract_faces(frame)[0]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (86, 32, 251), 1)
            cv2.rectangle(frame, (x, y), (x+w, y-40), (86, 32, 251), -1)
            face = cv2.resize(frame[y:y+h, x:x+w], (50, 50))
            identified_person = identify_face(face.reshape(1, -1))[0]
            add_attendance(identified_person)
            cv2.putText(frame, f'{identified_person}', (x+5, y-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow('Attendance', frame)
        if cv2.waitKey(1) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    names, rolls, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)


# A function to add a new user.
# This function will run when we add a new user.
@app.route('/add', methods=['GET', 'POST'])
def add():
    newusername = request.form['newusername']
    newuserid = request.form['newuserid']
    userimagefolder = 'static/faces/'+newusername+'_'+str(newuserid)
    if not os.path.isdir(userimagefolder):
        os.makedirs(userimagefolder)
    i, j = 0, 0
    cap = cv2.VideoCapture(0)
    while 1:
        _, frame = cap.read()
        faces = extract_faces(frame)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 20), 2)
            cv2.putText(frame, f'Images Captured: {i}/{nimgs}', (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2, cv2.LINE_AA)
            if j % 5 == 0:
                name = newusername+'_'+str(i)+'.jpg'
                cv2.imwrite(userimagefolder+'/'+name, frame[y:y+h, x:x+w])
                i += 1
            j += 1
        if j == nimgs*5:
            break
        cv2.imshow('Adding new User', frame)
        if cv2.waitKey(1) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    print('Training Model')
    train_model()
    names, rolls, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)


# Our main function which runs the Flask App
if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
