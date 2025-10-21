// --- Sign Out Logic & User Greeting ---
document.addEventListener('DOMContentLoaded', function() {
    // Set user greeting if on dashboard
    const userGreeting = document.getElementById('user-greeting');
    if (userGreeting) {
        const user = JSON.parse(localStorage.getItem('templevision_user') || 'null');
        if (user && user.name) {
            userGreeting.textContent = `Welcome, ${user.name}`;
        } else {
            userGreeting.textContent = 'Welcome, Researcher';
        }
    }
    // Sign out logic
    const signoutBtn = document.getElementById('signout-btn');
    if (signoutBtn) {
        signoutBtn.onclick = function() {
            localStorage.removeItem('templevision_user');
            window.location.href = 'login.html';
        };
    }
});
// --- Authentication Logic ---
document.addEventListener('DOMContentLoaded', function() {
    // If on login.html, set up login/signup logic
    if (window.location.pathname.endsWith('login.html')) {
    const loginForm = document.getElementById('loginForm');
    const signupForm = document.getElementById('signupForm');
        const showSignup = document.getElementById('show-signup');
        const showLogin = document.getElementById('show-login');
        const loginSection = document.getElementById('login-form');
        const signupSection = document.getElementById('signup-form');

        showSignup.onclick = function(e) {
            e.preventDefault();
            loginSection.style.display = 'none';
            signupSection.style.display = 'block';
        };
        showLogin.onclick = function(e) {
            e.preventDefault();
            signupSection.style.display = 'none';
            loginSection.style.display = 'block';
        };

        // Login with backend API
        loginForm.onsubmit = async function(e) {
            e.preventDefault();
            const email = document.getElementById('login-email').value.trim();
            const password = document.getElementById('login-password').value;
            const errorDiv = document.getElementById('login-error');
            errorDiv.textContent = '';
            try {
                const res = await fetch('http://localhost:4000/api/v1/users/login', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ email, password })
                });
                const data = await res.json();
                if (res.ok) {
                    localStorage.setItem('templevision_user', JSON.stringify(data.user));
                    localStorage.setItem('templevision_token', data.accessToken);
                    window.location.href = 'index.html';
                } else {
                    errorDiv.textContent = data.message || 'Login failed';
                }
            } catch (err) {
                errorDiv.textContent = 'Error: ' + err.message;
            }
        };

        // Signup with backend API
        signupForm.onsubmit = async function(e) {
            e.preventDefault();
            const name = document.getElementById('signup-name').value.trim();
            const email = document.getElementById('signup-email').value.trim();
            const password = document.getElementById('signup-password').value;
            const confirm = document.getElementById('signup-confirm').value;
            const errorDiv = document.getElementById('signup-error');
            errorDiv.textContent = '';
            if (!name || !email || !password || !confirm) {
                errorDiv.textContent = 'All fields are required.';
                return;
            }
            if (password !== confirm) {
                errorDiv.textContent = 'Passwords do not match.';
                return;
            }
            const body = { fullName: name, email, password };
            try {
                const res = await fetch('http://localhost:4000/api/v1/users/register', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(body)
                });
                const data = await res.json();
                if (res.ok) {
                    localStorage.setItem('templevision_user', JSON.stringify(data.user));
                    localStorage.setItem('templevision_token', data.accessToken);
                    window.location.href = 'index.html';
                } else {
                    errorDiv.textContent = data.message || 'Signup failed';
                }
            } catch (err) {
                errorDiv.textContent = 'Error: ' + err.message;
            }
        };
    }
});
document.getElementById('imageForm').onsubmit = async function(e) {
    e.preventDefault();
    const formData = new FormData(this);
    const resultDiv = document.getElementById('result');
    resultDiv.innerHTML = "Processing image...";
    try {
        const res = await fetch('http://localhost:5000/detect-image', {
            method: 'POST',
            body: formData
        });
        if (res.ok) {
            const blob = await res.blob();
            const url = URL.createObjectURL(blob);
            resultDiv.innerHTML = `<h2>Detected Image</h2><img src="${url}" alt="Result Image">`;
        } else {
            const err = await res.json();
            resultDiv.innerHTML = `<p style="color:red;">${err.error}</p>`;
        }
    } catch (error) {
        resultDiv.innerHTML = `<p style="color:red;">Error: ${error.message}</p>`;
    }
};

document.getElementById('videoForm').onsubmit = async function(e) {
    e.preventDefault();
    const formData = new FormData(this);
    const resultDiv = document.getElementById('result');
    resultDiv.innerHTML = "Processing video...";
    try {
        const res = await fetch('http://localhost:5000/detect-video', {
            method: 'POST',
            body: formData
        });
        if (res.ok) {
            const data = await res.json();
            resultDiv.innerHTML = `
                <h2>Detected Video</h2>
                <video controls src="http://localhost:5000/get-result-video"></video>
                <h3>Summary</h3>
                <pre>${JSON.stringify(data.summary, null, 2)}</pre>
            `;
        } else {
            const err = await res.json();
            resultDiv.innerHTML = `<p style="color:red;">${err.error}</p>`;
        }
    } catch (error) {
        resultDiv.innerHTML = `<p style="color:red;">Error: ${error.message}</p>`;
    }
};