import { async, ComponentFixture, TestBed } from '@angular/core/testing';

import { BenchmarkStepComponent } from './benchmark-step.component';

describe('BenchmarkStepComponent', () => {
  let component: BenchmarkStepComponent;
  let fixture: ComponentFixture<BenchmarkStepComponent>;

  beforeEach(async(() => {
    TestBed.configureTestingModule({
      declarations: [ BenchmarkStepComponent ]
    })
    .compileComponents();
  }));

  beforeEach(() => {
    fixture = TestBed.createComponent(BenchmarkStepComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
