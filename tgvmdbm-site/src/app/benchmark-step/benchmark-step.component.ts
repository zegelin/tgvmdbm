import { Component, OnInit, Input } from '@angular/core';

@Component({
  selector: 'app-benchmark-step',
  templateUrl: './benchmark-step.component.html',
  styleUrls: ['./benchmark-step.component.css']
})
export class BenchmarkStepComponent implements OnInit {
  @Input() step: any;

  constructor() { }

  ngOnInit() {
  }

}
