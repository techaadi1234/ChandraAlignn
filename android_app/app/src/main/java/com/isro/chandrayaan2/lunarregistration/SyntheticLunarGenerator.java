package com.isro.chandrayaan2.lunarregistration;

import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import java.util.Random;

/**
 * Java procedural generator for synthetic lunar crater surfaces.
 * Creates synthetic lunar pairs with shadow illumination shifts and perspective tilt.
 */
public class SyntheticLunarGenerator {

    public static class LunarPair {
        public Bitmap refBitmap;
        public Bitmap srcBitmap;

        public LunarPair(Bitmap refBitmap, Bitmap srcBitmap) {
            this.refBitmap = refBitmap;
            this.srcBitmap = srcBitmap;
        }
    }

    public static LunarPair generatePair(int width, int height, float sunAngle1Deg, float sunAngle2Deg) {
        Bitmap ref = generateSurface(width, height, sunAngle1Deg, 42);
        Bitmap src = generateSurface(width, height, sunAngle2Deg, 42);
        
        // Apply geometric transform (scale & rotation) to source image
        Bitmap srcWarped = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
        Canvas canvas = new Canvas(srcWarped);
        canvas.save();
        canvas.rotate(8.0f, width / 2.0f, height / 2.0f);
        canvas.scale(1.05f, 1.05f, width / 2.0f, height / 2.0f);
        canvas.drawBitmap(src, 0, 0, null);
        canvas.restore();

        return new LunarPair(ref, srcWarped);
    }

    private static Bitmap generateSurface(int width, int height, float sunAngleDeg, long seed) {
        Bitmap bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
        Canvas canvas = new Canvas(bitmap);
        canvas.drawColor(Color.rgb(120, 120, 120));

        Random rand = new Random(seed);
        Paint paintRim = new Paint();
        paintRim.setAntiAlias(true);

        double rad = Math.toRadians(sunAngleDeg);
        float shadowDx = (float) (Math.cos(rad) * 15.0);
        float shadowDy = (float) (Math.sin(rad) * 15.0);

        // Generate procedural craters
        for (int i = 0; i < 35; i++) {
            float cx = rand.nextFloat() * (width - 100) + 50;
            float cy = rand.nextFloat() * (height - 100) + 50;
            float r = rand.nextFloat() * 45 + 15;

            // Shadow interior
            paintRim.setColor(Color.rgb(40, 40, 40));
            canvas.drawCircle(cx - shadowDx * 0.3f, cy - shadowDy * 0.3f, r, paintRim);

            // Sunlit rim
            paintRim.setColor(Color.rgb(200, 200, 200));
            canvas.drawCircle(cx + shadowDx * 0.3f, cy + shadowDy * 0.3f, r * 0.85f, paintRim);

            // Crater floor
            paintRim.setColor(Color.rgb(100, 100, 100));
            canvas.drawCircle(cx, cy, r * 0.75f, paintRim);
        }

        return bitmap;
    }
}
