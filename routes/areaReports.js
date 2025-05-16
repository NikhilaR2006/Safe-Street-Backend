const express = require('express');
const router = express.Router();
const Upload = require('../models/Upload'); // Adjust path if needed

// GET /api/area-reports
router.get('/', async (req, res) => {
  try {
    const areaWiseData = await Upload.aggregate([
      {
        $group: {
          _id: "$pincode",
          uploads: { $push: "$$ROOT" }
        }
      },
      {
        $sort: { _id: 1 } // Sort by pincode ascending
      }
    ]);
    res.status(200).json(areaWiseData);
  } catch (err) {
    res.status(500).json({ message: 'Server Error', error: err.message });
  }
});

module.exports = router;
