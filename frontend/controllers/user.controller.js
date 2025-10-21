import { User } from "../models/user.model.js";

export const registerUser = async (req, res) => {
  try {
    const { fullName, email, password } = req.body;
    console.log("Register attempt:", { fullName, email });
    const existingUser = await User.findOne({ email });
    if (existingUser) {
      console.log("Email already registered:", email);
      return res.status(400).json({ message: "Email already registered" });
    }

    const user = await User.create({ fullName, email, password });
    console.log("User created:", user);
    const accessToken = user.generateAccessToken();
    res.status(201).json({ user: { fullName: user.fullName, email: user.email }, accessToken });
  } catch (err) {
    console.error("Registration failed:", err);
    res.status(500).json({ message: "Registration failed", error: err.message });
  }
};

export const loginUser = async (req, res) => {
  try {
    const { email, password } = req.body;
    console.log("Login attempt:", { email });
    const user = await User.findOne({ email });
    if (!user) {
      console.log("User not found:", email);
      return res.status(400).json({ message: "Invalid email or password" });
    }

    const isMatch = await user.isPasswordCorrect(password);
    if (!isMatch) {
      console.log("Password mismatch for:", email);
      return res.status(400).json({ message: "Invalid email or password" });
    }

    console.log("Login successful for:", email);
    const accessToken = user.generateAccessToken();
    res.status(200).json({ user: { fullName: user.fullName, email: user.email }, accessToken });
  } catch (err) {
    console.error("Login failed:", err);
    res.status(500).json({ message: "Login failed", error: err.message });
  }
};