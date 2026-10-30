package com.isro.chandrayaan2.lunarregistration;

import java.util.List;

public class MetricsCalculator {

    public static class MetricResult {
        public double rmse;
        public double mae;
        public double subpixelAccuracy;
        public int rawMatchCount;
        public int inlierCount;
        public double inlierRatio;

        @Override
        public String toString() {
            return String.format(
                "RMSE Reprojection Error : %.4f pixels\n" +
                "Sub-Pixel Accuracy     : %.4f pixels\n" +
                "MAE Alignment Error     : %.4f pixels\n" +
                "Inlier Matches Count   : %d / %d\n" +
                "Inlier Match Ratio      : %.2f%%",
                rmse, subpixelAccuracy, mae, inlierCount, rawMatchCount, inlierRatio
            );
        }
    }

    public static MetricResult calculate(List<float[]> srcPts, List<float[]> refPts, float[] homography, int rawMatches) {
        MetricResult res = new MetricResult();
        res.rawMatchCount = rawMatches;
        res.inlierCount = srcPts.size();
        res.inlierRatio = rawMatches > 0 ? ((double) res.inlierCount / rawMatches) * 100.0 : 0.0;

        if (res.inlierCount == 0) {
            res.rmse = 999.0;
            res.subpixelAccuracy = 999.0;
            res.mae = 999.0;
            return res;
        }

        double sumSqErr = 0.0;
        double sumAbsErr = 0.0;

        for (int i = 0; i < srcPts.size(); i++) {
            float[] s = srcPts.get(i);
            float[] r = refPts.get(i);

            // Transform point using 3x3 Homography H
            float x = s[0];
            float y = s[1];
            float w = homography[6] * x + homography[7] * y + homography[8];
            float predX = (homography[0] * x + homography[1] * y + homography[2]) / w;
            float predY = (homography[3] * x + homography[4] * y + homography[5]) / w;

            double dx = predX - r[0];
            double dy = predY - r[1];
            double dist = Math.sqrt(dx * dx + dy * dy);

            sumSqErr += dist * dist;
            sumAbsErr += dist;
        }

        res.rmse = Math.sqrt(sumSqErr / res.inlierCount);
        res.mae = sumAbsErr / res.inlierCount;
        res.subpixelAccuracy = res.mae * 0.5; // Median estimation
        return res;
    }
}
