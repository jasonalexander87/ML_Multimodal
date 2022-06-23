package com.example.pytorch;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Context;
import android.os.Bundle;
import android.util.Log;
import android.widget.TextView;

import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

public class MainActivity extends AppCompatActivity {

    TextView outp;
    Module module;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        outp=findViewById(R.id.hw);

        // Load the model
        try {
             module = Module.load(assetFilePath(this, "model.ptl"));
            Log.d("INFO","SUCCESS");
        } catch (IOException e) {
            e.printStackTrace();
            Log.d("INFO","FAILURE");
        }

        // Create input
        float[] a = new float[]{1.0f};
        Tensor input = Tensor.fromBlob(a,new long[]{1,1});

        //Run inference and get output
        Tensor outputTensor = module.forward(IValue.from(input)).toTensor();
        float[] scores = outputTensor.getDataAsFloatArray();

        // Display Result
        outp.setText(scores.toString());


    }

    public static String assetFilePath(Context context, String assetName) throws IOException {
        File file = new File(context.getFilesDir(), assetName);
        if (file.exists() && file.length() > 0) {
            return file.getAbsolutePath();
        }

        try (InputStream is = context.getAssets().open(assetName)) {
            try (OutputStream os = new FileOutputStream(file)) {
                byte[] buffer = new byte[4 * 1024];
                int read;
                while ((read = is.read(buffer)) != -1) {
                    os.write(buffer, 0, read);
                }
                os.flush();
            }
            return file.getAbsolutePath();
        }


    }
}