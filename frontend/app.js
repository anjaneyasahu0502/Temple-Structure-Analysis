import express from "express";
import cors from "cors";
import cookieParser from "cookie-parser";
import path from "path";

const app = express();

// Use CORS with credentials and origin from env
import dotenv from "dotenv";
dotenv.config({ path: "./.env" });
app.use(cors({
	origin: process.env.CORS_ORIGIN || "*",
	credentials: true
}));

app.use(express.json({ limit: "16kb" }));
app.use(express.urlencoded({ extended: true, limit: "16kb" }));

// Serve static files from the frontend directory
const __dirname = path.resolve();
app.use(express.static(__dirname));
app.use(cookieParser());

// routes import
import userRouter from "./routes/user.routes.js";

// routes declaration
app.use("/api/v1/users", userRouter);

export { app };