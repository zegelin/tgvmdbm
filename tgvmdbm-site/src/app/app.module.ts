import { BrowserModule } from '@angular/platform-browser';
import { NgModule } from '@angular/core';
import { HttpClientModule }    from '@angular/common/http';

import { AppComponent } from './app.component';
import { BenchmarksComponent } from './benchmarks/benchmarks.component';
import { BenchmarkComponent } from './benchmark/benchmark.component';
import { BenchmarkStepComponent } from './benchmark-step/benchmark-step.component';

@NgModule({
  declarations: [
    AppComponent,
    BenchmarksComponent,
    BenchmarkComponent,
    BenchmarkStepComponent
  ],
  imports: [
    BrowserModule,
    HttpClientModule
  ],
  providers: [],
  bootstrap: [AppComponent]
})
export class AppModule { }
