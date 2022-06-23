package com.example.tflite;

import androidx.appcompat.app.AppCompatActivity;

import android.content.res.AssetFileDescriptor;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;

import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;

public class MainActivity2 extends AppCompatActivity {
    Interpreter tflite;
    EditText inp;
    TextView outp;
    Button pred;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        pred=findViewById(R.id.bt);
        outp=findViewById(R.id.hw);
        inp=findViewById(R.id.et);

        //Create model variable
        try {
            tflite = new Interpreter(loadModelFile());
        }catch (Exception ex){
            ex.printStackTrace();
        }

        pred.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                float prediction=doInference(inp.getText().toString());
                System.out.println(prediction);
                outp.setText(Float.toString(prediction));
            }
        });

    }

    // Load model
    private MappedByteBuffer loadModelFile() throws IOException {
        AssetFileDescriptor fileDescriptor=this.getAssets().openFd("model.tflite");
        FileInputStream inputStream=new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel=inputStream.getChannel();
        long startOffset=fileDescriptor.getStartOffset();
        long declareLength=fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY,startOffset,declareLength);
    }
    //Do inference
    private float doInference(String inputString) {
        float[] inputVal=new float[1];
        inputVal[0]=Float.parseFloat(inputString);
        float[][] output=new float[1][1];
        tflite.run(inputVal,output);
        return output[0][0];
    }

}