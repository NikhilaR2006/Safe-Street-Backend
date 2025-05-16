const mongoose = require("mongoose");

const UserSchema = new mongoose.Schema({
    firstName: { type: String, required: true, trim: true },
    lastName: { type: String, required: true, trim: true },
    phoneNumber: { type: String, trim: true, default: '' },
    email: { type: String, required: true, unique: true, trim: true, lowercase: true },
    password: { type: String, required: true },
    address: { type: String, trim: true, default: '' },
    district: { type: String, trim: true, default: '' },
    pincode: { type: String, trim: true, default: '' },
    role: { type: String, required: true, enum: ['user', 'authority'], default: 'user' },
    createdAt: { type: Date, default: Date.now }
});

module.exports = mongoose.model("User", UserSchema);
444