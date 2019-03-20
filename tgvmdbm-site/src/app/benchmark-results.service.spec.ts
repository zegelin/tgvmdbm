import { TestBed } from '@angular/core/testing';

import { BenchmarkResultsService } from './benchmark-results.service';

describe('BenchmarkResultsService', () => {
  beforeEach(() => TestBed.configureTestingModule({}));

  it('should be created', () => {
    const service: BenchmarkResultsService = TestBed.get(BenchmarkResultsService);
    expect(service).toBeTruthy();
  });
});
