const mongoose = require("mongoose");

const NotificationSchema = new mongoose.Schema({
    userEmail: { type: String, required: true },
    type: { type: String, required: true }, // 'resolved', 'status_update', etc.
    message: { type: String, required: true },
    reportId: { type: mongoose.Schema.Types.ObjectId, ref: 'RoadImage' },
    read: { type: Boolean, default: false },
    createdAt: { type: Date, default: Date.now }
});

module.exports = mongoose.model("Notification", NotificationSchema);
