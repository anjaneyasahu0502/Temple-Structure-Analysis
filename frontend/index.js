import connectDB from './db/index.js';
import { app } from "./app.js";   // ✅ import the configured app
import dotenv from "dotenv";
dotenv.config({ path: "./env" });  //try ./.env

connectDB()
  .then(() => {
    app.listen(process.env.PORT || 4000, () => {
      console.log(`🚀 Server is running at port: ${process.env.PORT || 4000}`);
    });
  })
  .catch((err) => {
    console.log("❌ MONGO DB connection failed !!", err);
  });