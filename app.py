from flask import Flask, render_template, Response, redirect, url_for, request
from flask_login import LoginManager, login_user, login_required, logout_user, UserMixin
from detector import generate_frames

app = Flask(__name__)
app.secret_key = 'secret-key'

login_manager = LoginManager()
login_manager.init_app(app)

# Dummy admin user
class User(UserMixin):
    def __init__(self, id):
        self.id = id

users = {'admin': {'password': 'admin123'}}

@login_manager.user_loader
def load_user(user_id):
    return User(user_id)

@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        if users.get(request.form['username']) and users[request.form['username']]['password'] == request.form['password']:
            user = User(request.form['username'])
            login_user(user)
            return redirect(url_for('dashboard'))
        return "Invalid credentials"
    return render_template('login.html')

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html')

@app.route('/video_feed')
@login_required
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)
