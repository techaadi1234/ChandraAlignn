package com.isro.chandrayaan2.lunarregistration;

import android.graphics.Bitmap;
import android.os.Bundle;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;
import androidx.appcompat.app.AppCompatActivity;

/**
 * Main Activity for Chandrayaan-2 Lunar Image Registration Android App.
 */
public class MainActivity extends AppCompatActivity {

    private ImageView imgReference, imgSource, imgRegistered;
    private TextView txtMetrics;
    private Button btnGenerateSynthetic, btnRunRegistration;

    private SyntheticLunarGenerator.LunarPair lunarPair;
    private LunarRegistrarEngine.RegistrationResult regResult;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        imgReference = findViewById(R.id.imgReference);
        imgSource = findViewById(R.id.imgSource);
        imgRegistered = findViewById(R.id.imgRegistered);
        txtMetrics = findViewById(R.id.txtMetrics);
        btnGenerateSynthetic = findViewById(R.id.btnGenerateSynthetic);
        btnRunRegistration = findViewById(R.id.btnRunRegistration);

        // 1. Generate Synthetic Pair Button Listener
        btnGenerateSynthetic.setOnClickListener(v -> generateSyntheticPair());

        // 2. Run Registration Engine Button Listener
        btnRunRegistration.setOnClickListener(v -> runRegistrationPipeline());

        // Auto generate on launch
        generateSyntheticPair();
    }

    private void generateSyntheticPair() {
        lunarPair = SyntheticLunarGenerator.generatePair(400, 400, 35.0f, 55.0f);
        imgReference.setImageBitmap(lunarPair.refBitmap);
        imgSource.setImageBitmap(lunarPair.srcBitmap);
        imgRegistered.setImageDrawable(null);
        txtMetrics.setText("Synthetic Lunar Pair generated.\nClick 'Run Sub-Pixel Registration' to align.");
        Toast.makeText(this, "Synthetic Lunar Crater Pair Generated!", Toast.LENGTH_SHORT).show();
    }

    private void runRegistrationPipeline() {
        if (lunarPair == null || lunarPair.refBitmap == null || lunarPair.srcBitmap == null) {
            Toast.makeText(this, "Please generate or load lunar images first.", Toast.LENGTH_SHORT).show();
            return;
        }

        regResult = LunarRegistrarEngine.register(lunarPair.srcBitmap, lunarPair.refBitmap);
        imgRegistered.setImageBitmap(regResult.registeredBitmap);

        if (regResult.metrics != null) {
            txtMetrics.setText(regResult.metrics.toString());
        }
        Toast.makeText(this, "Sub-Pixel Lunar Alignment Completed!", Toast.LENGTH_SHORT).show();
    }
}
